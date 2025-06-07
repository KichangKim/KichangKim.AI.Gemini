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

while (true)
{
    Console.Write("You> ");
    var input = Console.ReadLine();
    Console.WriteLine();
    
    var response = await client.GetResponseAsync(new ChatMessage(ChatRole.User, input), chatOptions);

    Console.Write("AI> ");
    Console.WriteLine(response.Text);
}
```
- Streaming response
```csharp
using KichangKim.AI.Gemini;
using Microsoft.Extensions.AI;

var apiKey = "[YOUR GEMINI API KEY]";
var model = "[YOUR GEMINI MODEL NAME]";

var client = new GeminiChatClient(apiKey, model, includeThoughts: false, thinkingBudget: 0);

// For disable thinking
var chatOptions = new ChatOptions()
{
    AdditionalProperties = new AdditionalPropertiesDictionary()
    {
        [GeminiChatClient.ThinkingBudgetKey] = 0,
        [GeminiChatClient.IncludeThoughtsKey] = false,
    },
};

while (true)
{
    Console.Write("You> ");
    var input = Console.ReadLine();
    Console.WriteLine();

    Console.Write("AI> ");

    await foreach (var update in client.GetStreamingResponseAsync(new[] { new ChatMessage(ChatRole.User, input) }, chatOptions))
    {
        Console.Write(update.Text);
    }

    Console.WriteLine();
}
```
