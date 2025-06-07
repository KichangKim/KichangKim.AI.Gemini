# KichangKim.AI.Gemini
This library contains `GeminiChatClient` for Google Gemini API. It implements `Microsoft.Extensions.AI.IChatClient` and supports function call and structured output, so you can use 
`UseFunctionInvocation()` extensions. Also `GeminiChatClient` provides settings related to thinking which is available on gemini-2.5-flash model.

Last, most of this library is written by gemini-2.5-pro-preview-06-05 model :), I fixed some of errors and changed method signatures.

# Usage
- Single response
```csharp
using KichangKim.AI.Gemini;
using Microsoft.Extensions.AI;

var apiKey = "[YOUR GEMINI API KEY]";
var model = "[YOUR GEMINI MODEL NAME]";

var client = new GeminiChatClient(apiKey, model);

// For disable thinking
var chatOptions = new ChatOptions()
{
    AdditionalProperties = new AdditionalPropertiesDictionary()
    {
        [GeminiChatClient.ThinkingBudgetKey] = 0,
        [GeminiChatClient.IncludeThoughtsKey] = false,
    },
};

var query = "What is 1 + 1 (spelling out numbers)";
Console.WriteLine($"Q: {query}");

var response = await client.GetResponseAsync(query, chatOptions);
Console.Write($"A: {response.Text}");
```
- Streaming response
```csharp
var client = new GeminiChatClient(apiKey, model);

var query = "Say long text.";
Console.WriteLine($"Q: {query}");

Console.Write($"A: ");
await foreach (var update in client.GetStreamingResponseAsync(query))
{
    Console.Write(update.Text);
}
```
- Structured output
```csharp
var client = new GeminiChatClient(apiKey, model);

var question = "What is 1 + 1?";
Console.WriteLine($"Q: {question}");

var response = await client.GetResponseAsync<int>(question);
Console.WriteLine($"A : {response.Result}");
```
