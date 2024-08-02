use std::env;
use lopdf::Document;
use qdrant_client::qdrant::Query;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::de::Error;
use serde_json::{json, Value};
use serde::{Serialize, Deserialize};

// it should be serializable
#[derive(serde::Serialize)]
pub struct EmbeddingRequest {
    input: String,
    model: String,
}

#[derive(serde::Deserialize)]
pub struct EmbeddingData {
    embedding: Vec<f64>
}

#[derive(serde::Deserialize)]
pub struct EmbeddingResponse {
    data: Vec<EmbeddingData>
}


pub fn extract_text_from_pdf(file_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let doc = Document::load(file_path)?;
    let mut texts: Vec<String> = Vec::new();

    let pages = doc.get_pages();

    for (i, _) in pages.iter().enumerate() {
        let page_number = (i + 1) as u32;
        let text = doc.extract_text(&[page_number]);

        match text {
            Ok(page_text) => {
                for line in page_text.lines() {
                    if line.len() > 512 {
                        texts.push(line[..512].to_string());
                        texts.push(line[512..].to_string());
                    } else {
                        texts.push(line.to_string());
                    }
                }
            }
            Err(_) => {
                texts.push("".to_string());
            }
        }
    }

    Ok(texts)
}

pub async fn store_pdf_vectorstore(client: Client, file_path: &str) {

    // pinecone,
    let api_key = env::var("PINECONE_API_KEY").expect("PINECONE_API_KEY not set");
    let pinecone_index_host = env::var("PINECONE_URL").expect("PINECONE_INDEX_HOST not set");
    let client = Client::new();

    let mut pc = PineConeClient{
        api_key: api_key,
        index_host: pinecone_index_host,
        client: client
    };

    // read pdf from knowledge base
    let text_by_pages = extract_text_from_pdf("./knowledge-base/Tanmay_Sharma.pdf")
        .expect("Error reading the pdf file!");

    let mut pdf_embeddings: Vec<EmbeddingObject> = Vec::new();

    for text in text_by_pages {
        // generate vector embeddings
        let text_embeddings = get_embedding(text.as_str()).await.unwrap_or_else(|err| {
            eprintln!("Error generating embeddings: {}", err);
            Vec::new()
        });

        // create EmbeddingObject and push it to the pdf_embeddings
        let embedding_object = EmbeddingObject{
            id: text,
            embedding: text_embeddings
        };

        pdf_embeddings.push(embedding_object);
    }

    pc.upsert_vectors("ns1", pdf_embeddings).await;
}

async fn gen_query_embedding(query: &str) -> Vec<f64> {

    let query_embedding = get_embedding(query).await.unwrap_or_else(|err| {
        eprintln!("Error generating embeddings: {}", err);
        Vec::new()
    });

    query_embedding
}

pub async fn query_vector_store(namespace: &str, query: &str) -> Vec<Value> {

    // pinecone,
    let api_key = env::var("PINECONE_API_KEY").expect("PINECONE_API_KEY not set");
    let pinecone_index_host = env::var("PINECONE_URL").expect("PINECONE_INDEX_HOST not set");
    let client = Client::new();

    let mut pc = PineConeClient{
        api_key: api_key,
        index_host: pinecone_index_host,
        client: client
    };

    let query_embedding = gen_query_embedding(query).await;

    let query_matches = pc.query_vector(namespace, query_embedding, 5, false).await;

    query_matches

}

pub async fn store_text_vector_store(namespace: &str, text: &str) {

    // pinecone,
    let api_key = env::var("PINECONE_API_KEY").expect("PINECONE_API_KEY not set");
    let pinecone_index_host = env::var("PINECONE_URL").expect("PINECONE_INDEX_HOST not set");
    let client = Client::new();

    let mut pc = PineConeClient{
        api_key: api_key,
        index_host: pinecone_index_host,
        client: client
    };

    let text_embedding = gen_query_embedding(text).await;
    let embedding_obj = EmbeddingObject {
        id: text.parse().unwrap(),
        embedding: text_embedding
    };

    pc.upsert_vectors(namespace, vec![embedding_obj]).await;
}



pub async fn get_embedding(text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {

    let openai_api_key = env::var("OPENAI_API_KEY").unwrap();

    let client = reqwest::Client::new();
    let request = EmbeddingRequest {
        input: text.to_string(),
        model: "text-embedding-ada-002".to_string()
    };

    let response = client
        .post("https://api.openai.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", openai_api_key))
        .json(&request)
        .send()
        .await.unwrap();

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await?;
        return Err(format!("Request failed with status {}: {}", status, text).into());
    }

    let response_data: EmbeddingResponse = response.json().await?;

    Ok(response_data.data[0].embedding.clone())

}


#[derive(Serialize, Deserialize)]
pub struct LLMMessage {
    pub role: String,
    pub content: String
}

pub async fn chat_completion(messages: Vec<LLMMessage>) -> Result<(), Box<dyn std::error::Error>> {
    // Get the OpenAI API key from the environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    // Set up the headers
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key))?);

    // Create the JSON body
    let body = json!({
        "model": "gpt-3.5-turbo",
        "messages": messages
    });

    // Send the request
    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .headers(headers)
        .json(&body)
        .send()
        .await?;

    // Print the response
    let res_text = res.text().await?;
    println!("{}", res_text);

    Ok(())
}


pub struct PineConeClient {
    pub api_key: String,
    pub index_host: String,
    pub client: Client,
}

#[derive(Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub id: String,
    pub embedding: Vec<f64>
}

// pub struct QueryResponse {
//     matches: vec![],
//     namespace: String,
//     usage:
// }


impl PineConeClient {
    pub async fn create_index(&self) {
        let url = "https://api.pinecone.io/indexes";

        let body = json!({
        "name": "resume-collection",
        "dimension": 1536,
        "metric": "cosine",
        "spec": {
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    });

        let response = self.client.post(url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("Api-Key", self.api_key.clone())
            .json(&body)
            .send()
            .await
            .expect("Failed to send request");

        if response.status().is_success() {
            println!("Index created successfully!");
        } else {
            println!("Failed to create index: {:?}", response.text().await);
        }
    }


    pub async fn upsert_vectors(
        &self,
        namespace: &str,
        embedding_vectors: Vec<EmbeddingObject>
    ) {
        let url = format!("https://{}/vectors/upsert", self.index_host);

        let vectors: Vec<serde_json::Value> = embedding_vectors.into_iter().map(|obj| {
            json!({
                "id": obj.id,
                "values": obj.embedding
            })
        }).collect();

        let body = json!({
            "vectors": vectors,
            "namespace": namespace
        });

        let response = self.client.post(&url)
            .header("Api-Key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .expect("Failed to send request");

        if response.status().is_success() {
            println!("Vectors upserted successfully to namespace {}", namespace);
        } else {
            println!("Failed to upsert vectors to namespace {}: {:?}", namespace, response.text().await);
        }
    }

    pub async fn query_vector(
        &self,
        namespace: &str,
        vector: Vec<f64>,
        top_k: usize,
        include_values: bool,
    ) -> Vec<Value> {
        let url = format!("https://{}/query", self.index_host);

        let body = json!({
            "namespace": namespace,
            "vector": vector,
            "topK": top_k,
            "includeValues": include_values
        });

        let response = self.client.post(&url)
            .header("Api-Key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .expect("Failed to send request");

        if response.status().is_success() {
            let result: Value = response.json().await.expect("Failed to parse response");
            let query_matches = result.get("matches").unwrap().as_array().unwrap();
            query_matches.clone()
        } else {
            println!("Failed to query vector from namespace {}: {:?}", namespace, response.text().await);
            let empty_vec: Vec<Value> = Vec::new();
            empty_vec
        }
    }
}