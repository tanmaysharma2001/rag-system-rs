use std::fmt::format;
use serde_json::Value;
use crate::lib::{chat_completion, LLMMessage, query_vector_store, store_text_vector_store};

mod lib;


fn augment_prompt(query: &str, query_matches: Vec<Value>) -> String {
    let mut source_knowledge: String = String::from("");
    for query_match in query_matches {
        let query_match_string = query_match.get("id").unwrap().as_str().unwrap();
        source_knowledge.push_str(query_match_string);
        source_knowledge.push('\n');
    }

    let augmented_prompt = format!("Using the context below, answer the query.\n\nContext: {}\n\nQuery: {}\n", source_knowledge, query);

    augmented_prompt

}

async fn get_answer(query: &str) {
    let query_matches = query_vector_store("ns1", query).await;

    let augmented_prompt = augment_prompt(query, query_matches);

    let messages: Vec<LLMMessage> = vec![
        LLMMessage {
            role: "system".parse().unwrap(),
            content: "You are a helpful assistant!".parse().unwrap()
        },
        LLMMessage {
            role: String::from("user"),
            content: augmented_prompt
        }
    ];

    let response = chat_completion(messages).await;

}


#[tokio::main]
async fn main() {

    // store_text_vector_store("ns1",
    //                         "Tanmay likes to code low level stuff in Rust programming language because it is more efficient and memory safe.")
    //     .await;

    // query
    get_answer("What tanmay likes").await;

}
