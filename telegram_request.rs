use reqwest::Client;
use serde::Serialize;

// 1. THE TRANSLATOR: We define the exact shape of the data.
// The `#[derive(Serialize)]` macro tells Serde to automatically write 
// the code that converts this strict struct into a standard JSON object.
#[derive(Serialize)]
struct TelegramMessage {
    chat_id: String,
    text: String,
}

// 2. THE ENGINE: The `#[tokio::main]` macro sets up our async runtime,
// allowing our agent to do other things while waiting for Telegram to reply.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bot_token = "YOUR_BOT_TOKEN_HERE";
    let chat_id = "YOUR_CHAT_ID_HERE".to_string();
    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token);

    // Instantiate our strictly-typed payload
    let payload = TelegramMessage {
        chat_id,
        text: "Hello from my Rust AI Agent! 🦀".to_string(),
    };

    // 3. THE MESSENGER: Create the HTTP client
    let client = Client::new();
    
    // Send the POST request
    let response = client
        .post(&url)
        .json(&payload) // Serde silently translates the Rust struct to JSON right here
        .send()
        .await?;        // Await the network response asynchronously

    // Safely check if Telegram accepted our request
    if response.status().is_success() {
        println!("Message successfully delivered to Telegram!");
    } else {
        println!("Uh oh, Telegram rejected it: {:?}", response.text().await?);
    }

    Ok(())
}
