using Microsoft.Extensions.AI;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace KichangKim.AI.Gemini
{
#nullable disable

    /// <summary>
    /// A chat client for interacting with the Google Gemini API.
    /// This class implements the IChatClient interface and provides methods
    /// for sending and receiving chat messages, including support for streaming,
    //  function calling, and structured output.
    /// </summary>
    public sealed class GeminiChatClient : IChatClient, IDisposable
    {
        public static readonly string ThinkingBudgetKey = "generationConfig.thinkingConfig.thinkingBudget";
        public static readonly string IncludeThoughtsKey = "generationConfig.thinkingConfig.includeThoughts";

        private const string GeminiApiEndpoint = "https://generativelanguage.googleapis.com";
        private static readonly JsonSerializerOptions s_jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseUpper) }
        };

        private readonly HttpClient _httpClient;
        private readonly string _apiKey;
        private readonly string _modelName;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="GeminiChatClient"/> class.
        /// </summary>
        /// <param name="apiKey">The Google Gemini API key.</param>
        /// <param name="model">The name of the Gemini model to use.</param>
        public GeminiChatClient(string apiKey, string model)
        {
            if (string.IsNullOrWhiteSpace(apiKey))
                throw new ArgumentException("API key cannot be null or whitespace.", nameof(apiKey));
            if (string.IsNullOrWhiteSpace(model))
                throw new ArgumentException("Model cannot be null or whitespace.", nameof(model));

            _apiKey = apiKey;
            _modelName = model;
            _httpClient = new HttpClient { BaseAddress = new Uri(GeminiApiEndpoint) };
        }

        /// <inheritdoc/>
        public async Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions options = null,
            CancellationToken cancellationToken = default)
        {
            if (messages == null)
            {
                throw new ArgumentNullException(nameof(messages));
            }

            var request = BuildGeminiRequest(messages, options);
            var requestUri = $"/v1beta/models/{_modelName}:generateContent?key={_apiKey}";

            var jsonContent = JsonSerializer.Serialize(request, s_jsonOptions);
            using (var requestContent = new StringContent(jsonContent, Encoding.UTF8, "application/json"))
            using (var requestMessage = new HttpRequestMessage(HttpMethod.Post, requestUri) { Content = requestContent })
            {
                using (var responseMessage = await _httpClient.SendAsync(requestMessage, cancellationToken).ConfigureAwait(false))
                {
                    if (!responseMessage.IsSuccessStatusCode)
                    {
                        var errorContent = await responseMessage.Content.ReadAsStringAsync().ConfigureAwait(false);
                        throw new HttpRequestException($"Gemini API request failed with status code {responseMessage.StatusCode}: {errorContent}");
                    }

                    using (var responseStream = await responseMessage.Content.ReadAsStreamAsync().ConfigureAwait(false))
                    {
                        var geminiResponse = await JsonSerializer.DeserializeAsync<GeminiResponse>(responseStream, s_jsonOptions, cancellationToken).ConfigureAwait(false);
                        return ConvertGeminiResponseToChatResponse(geminiResponse);
                    }
                }
            }
        }

        /// <inheritdoc/>
        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (messages == null)
            {
                throw new ArgumentNullException(nameof(messages));
            }

            var request = BuildGeminiRequest(messages, options);
            var requestUri = $"/v1beta/models/{_modelName}:streamGenerateContent?key={_apiKey}&alt=sse";

            var jsonContent = JsonSerializer.Serialize(request, s_jsonOptions);
            using (var requestContent = new StringContent(jsonContent, Encoding.UTF8, "application/json"))
            using (var requestMessage = new HttpRequestMessage(HttpMethod.Post, requestUri) { Content = requestContent })
            {
                using (var responseMessage = await _httpClient.SendAsync(requestMessage, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false))
                {
                    if (!responseMessage.IsSuccessStatusCode)
                    {
                        var errorContent = await responseMessage.Content.ReadAsStringAsync().ConfigureAwait(false);
                        throw new HttpRequestException($"Gemini API stream request failed with status code {responseMessage.StatusCode}: {errorContent}");
                    }

                    using (var stream = await responseMessage.Content.ReadAsStreamAsync().ConfigureAwait(false))
                    using (var reader = new StreamReader(stream))
                    {
                        while (!reader.EndOfStream)
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            var line = await reader.ReadLineAsync().ConfigureAwait(false);
                            if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data: "))
                            {
                                continue;
                            }

                            var jsonData = line.Substring("data: ".Length);
                            var geminiResponse = JsonSerializer.Deserialize<GeminiResponse>(jsonData, s_jsonOptions);

                            foreach (var update in ConvertGeminiResponseToChatResponseUpdates(geminiResponse))
                            {
                                yield return update;
                            }
                        }
                    }
                }
            }
        }

        /// <inheritdoc/>
        public object GetService(Type serviceType, object serviceKey = null)
        {
            if (serviceType == null)
            {
                throw new ArgumentNullException(nameof(serviceType));
            }
            if (serviceType == typeof(GeminiChatClient) || serviceType == typeof(IChatClient))
            {
                return this;
            }
            return null;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _httpClient.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        // Recursively sanitizes a JsonElement representing a JSON Schema to remove unsupported keywords for Gemini.
        private JsonElement SanitizeSchemaForGemini(JsonElement schemaElement)
        {
            if (schemaElement.ValueKind != JsonValueKind.Object)
            {
                return schemaElement;
            }

            using var ms = new MemoryStream();
            using (var writer = new Utf8JsonWriter(ms))
            {
                writer.WriteStartObject();
                foreach (var property in schemaElement.EnumerateObject())
                {
                    // Gemini does not support the '$schema' keyword at any level.
                    if (property.NameEquals("$schema"))
                    {
                        continue;
                    }

                    // Recursively sanitize nested objects.
                    if (property.Value.ValueKind == JsonValueKind.Object)
                    {
                        writer.WritePropertyName(property.Name);
                        var sanitizedNested = SanitizeSchemaForGemini(property.Value);
                        sanitizedNested.WriteTo(writer);
                    }
                    else
                    {
                        property.WriteTo(writer);
                    }
                }
                writer.WriteEndObject();
            }

            ms.Position = 0;
            return JsonDocument.Parse(ms).RootElement.Clone();
        }

        // *** MODIFICATION START ***
        // Helper function to convert JSON type string to GeminiSchemaType enum
        private GeminiSchemaType? ConvertJsonTypeToGeminiSchemaType(string typeString) => typeString?.ToLowerInvariant() switch
        {
            "object" => GeminiSchemaType.Object,
            "array" => GeminiSchemaType.Array,
            "string" => GeminiSchemaType.String,
            "number" => GeminiSchemaType.Number,
            "integer" => GeminiSchemaType.Integer,
            "boolean" => GeminiSchemaType.Boolean,
            _ => null
        };

        // Helper function to parse a schema element into a GeminiSchema object
        private GeminiSchema ParsePropertySchema(JsonElement propertyElement)
        {
            var schema = new GeminiSchema();

            if (propertyElement.TryGetProperty("type", out var typeProp) && typeProp.ValueKind == JsonValueKind.String)
            {
                schema.Type = ConvertJsonTypeToGeminiSchemaType(typeProp.GetString());
            }

            if (propertyElement.TryGetProperty("description", out var descriptionProp) && descriptionProp.ValueKind == JsonValueKind.String)
            {
                schema.Description = descriptionProp.GetString();
            }

            // Handle array items
            if (schema.Type == GeminiSchemaType.Array && propertyElement.TryGetProperty("items", out var itemsProp) && itemsProp.ValueKind == JsonValueKind.Object)
            {
                schema.Items = ParsePropertySchema(itemsProp);
            }

            // Handle nested object properties
            if (schema.Type == GeminiSchemaType.Object && propertyElement.TryGetProperty("properties", out var propertiesProp) && propertiesProp.ValueKind == JsonValueKind.Object)
            {
                var geminiProperties = new Dictionary<string, GeminiSchema>();
                foreach (var subProperty in propertiesProp.EnumerateObject())
                {
                    geminiProperties[subProperty.Name] = ParsePropertySchema(subProperty.Value);
                }
                schema.Properties = JsonSerializer.SerializeToElement(geminiProperties, s_jsonOptions);
            }

            return schema;
        }
        // *** MODIFICATION END ***

        // Maps the abstract AI models to the Gemini-specific request model.
        private GeminiRequest BuildGeminiRequest(IEnumerable<ChatMessage> messages, ChatOptions options)
        {
            var messageList = messages.ToList();
            var systemMessage = messageList.FirstOrDefault(m => m.Role == ChatRole.System);

            var geminiContents = new List<GeminiContent>();
            var history = messageList.Where(m => m.Role != ChatRole.System);

            // Gemini requires alternating user/model roles. We merge consecutive messages from the same author.
            foreach (var message in history)
            {
                var role = ConvertRole(message.Role);
                var lastContent = geminiContents.LastOrDefault();

                if (lastContent != null && lastContent.Role == role)
                {
                    lastContent.Parts.AddRange(ConvertToGeminiParts(message.Contents));
                }
                else
                {
                    geminiContents.Add(new GeminiContent
                    {
                        Role = role,
                        Parts = ConvertToGeminiParts(message.Contents)
                    });
                }
            }

            var request = new GeminiRequest
            {
                Contents = geminiContents,
                SystemInstruction = systemMessage != null ? new GeminiSystemInstruction { Parts = ConvertToGeminiParts(systemMessage.Contents) } : null
            };

            if (options != null)
            {
                var config = new GeminiGenerationConfig();
                bool configHasValues = false;

                if (options.Temperature.HasValue) { config.Temperature = options.Temperature; configHasValues = true; }
                if (options.MaxOutputTokens.HasValue) { config.MaxOutputTokens = options.MaxOutputTokens; configHasValues = true; }
                if (options.TopP.HasValue) { config.TopP = options.TopP; configHasValues = true; }
                if (options.TopK.HasValue) { config.TopK = options.TopK; configHasValues = true; }
                if (options.StopSequences?.Any() == true) { config.StopSequences = options.StopSequences.ToList(); configHasValues = true; }

                if (options.ResponseFormat is ChatResponseFormatJson jsonFormat)
                {
                    config.ResponseMimeType = "application/json";
                    configHasValues = true;
                    if (jsonFormat.Schema.HasValue)
                    {
                        var sanitizedElement = SanitizeSchemaForGemini(jsonFormat.Schema.Value);
                        var rootSchema = new GeminiSchema();

                        // *** MODIFICATION START ***
                        // Correctly parse the entire schema structure instead of just assigning properties.
                        if (sanitizedElement.TryGetProperty("type", out var typeProp) && typeProp.ValueKind == JsonValueKind.String)
                        {
                            rootSchema.Type = ConvertJsonTypeToGeminiSchemaType(typeProp.GetString());
                        }

                        if (sanitizedElement.TryGetProperty("properties", out var propertiesProp) && propertiesProp.ValueKind == JsonValueKind.Object)
                        {
                            var geminiProperties = new Dictionary<string, GeminiSchema>();
                            foreach (var property in propertiesProp.EnumerateObject())
                            {
                                // For each property in the schema, parse it into a valid GeminiSchema object.
                                geminiProperties[property.Name] = ParsePropertySchema(property.Value);
                            }
                            // Serialize the reconstructed dictionary to JsonElement.
                            rootSchema.Properties = JsonSerializer.SerializeToElement(geminiProperties, s_jsonOptions);
                        }

                        if (sanitizedElement.TryGetProperty("required", out var requiredProp) && requiredProp.ValueKind == JsonValueKind.Array)
                        {
                            rootSchema.Required = requiredProp.EnumerateArray().Select(e => e.GetString()).ToList();
                        }

                        // Use the SchemaDescription from the options.
                        if (!string.IsNullOrEmpty(jsonFormat.SchemaDescription))
                        {
                            rootSchema.Description = jsonFormat.SchemaDescription;
                        }
                        else if (sanitizedElement.TryGetProperty("description", out var descProp) && descProp.ValueKind == JsonValueKind.String)
                        {
                            rootSchema.Description = descProp.GetString();
                        }

                        config.ResponseSchema = rootSchema;
                        // *** MODIFICATION END ***
                    }
                }

                if (options.AdditionalProperties != null)
                {
                    GeminiThinkingConfig thinkingConfig = null;

                    if (options.AdditionalProperties.TryGetValue(ThinkingBudgetKey, out object budgetValue))
                    {
                        int? parsedBudget = null;
                        if (budgetValue is int intVal) parsedBudget = intVal;
                        else if (budgetValue is long longVal) parsedBudget = (int)longVal;
                        else if (budgetValue is JsonElement jElem && jElem.TryGetInt32(out int jInt)) parsedBudget = jInt;

                        if (parsedBudget.HasValue)
                        {
                            thinkingConfig ??= new GeminiThinkingConfig();
                            thinkingConfig.ThinkingBudget = parsedBudget.Value;
                        }
                    }

                    if (options.AdditionalProperties.TryGetValue(IncludeThoughtsKey, out object thoughtsValue))
                    {
                        bool? parsedThoughts = null;
                        if (thoughtsValue is bool boolVal) parsedThoughts = boolVal;
                        else if (thoughtsValue is JsonElement jElem && jElem.ValueKind == JsonValueKind.True) parsedThoughts = true;
                        else if (thoughtsValue is JsonElement jElem2 && jElem2.ValueKind == JsonValueKind.False) parsedThoughts = false;

                        if (parsedThoughts.HasValue)
                        {
                            thinkingConfig ??= new GeminiThinkingConfig();
                            thinkingConfig.IncludeThoughts = parsedThoughts.Value;
                        }
                    }

                    if (thinkingConfig != null)
                    {
                        config.ThinkingConfig = thinkingConfig;
                        configHasValues = true;
                    }
                }

                if (configHasValues)
                {
                    request.GenerationConfig = config;
                }

                if (options.Tools?.Any() == true)
                {
                    var functionDeclarations = options.Tools.OfType<AIFunction>().Select(ConvertAIFunctionToGemini).ToList();
                    if (functionDeclarations.Any())
                    {
                        request.Tools = new List<GeminiTool> { new GeminiTool { FunctionDeclarations = functionDeclarations } };
                    }
                }

                if (options.ToolMode != null)
                {
                    request.ToolConfig = new GeminiToolConfig
                    {
                        FunctionCallingConfig = options.ToolMode switch
                        {
                            NoneChatToolMode => new GeminiFunctionCallingConfig { Mode = GeminiFunctionCallingMode.None },
                            AutoChatToolMode => new GeminiFunctionCallingConfig { Mode = GeminiFunctionCallingMode.Auto },
                            RequiredChatToolMode required => new GeminiFunctionCallingConfig
                            {
                                Mode = GeminiFunctionCallingMode.Any,
                                AllowedFunctionNames = string.IsNullOrEmpty(required.RequiredFunctionName) ? null : new List<string> { required.RequiredFunctionName }
                            },
                            _ => null
                        }
                    };
                }
            }
            return request;
        }

        // Converts a Gemini API response to a standard ChatResponse.
        private ChatResponse ConvertGeminiResponseToChatResponse(GeminiResponse geminiResponse)
        {
            var candidate = geminiResponse.Candidates?.FirstOrDefault();
            if (candidate == null)
            {
                var reason = geminiResponse.PromptFeedback?.BlockReason;
                throw new InvalidOperationException($"The request was blocked or returned no candidates. Reason: {(reason ?? "Unknown")}");
            }

            var responseMessage = new ChatMessage
            {
                Role = ChatRole.Assistant,
                Contents = ConvertGeminiPartsToAIContent(candidate.Content?.Parts)
            };

            var response = new ChatResponse
            {
                ModelId = _modelName,
                FinishReason = ConvertFinishReason(candidate.FinishReason, candidate.Content?.Parts),
                RawRepresentation = geminiResponse
            };
            response.Messages.Add(responseMessage);

            if (geminiResponse.UsageMetadata != null)
            {
                response.Usage = new UsageDetails
                {
                    InputTokenCount = geminiResponse.UsageMetadata.PromptTokenCount,
                    OutputTokenCount = geminiResponse.UsageMetadata.CandidatesTokenCount,
                    TotalTokenCount = geminiResponse.UsageMetadata.TotalTokenCount
                };
            }

            return response;
        }

        // Converts a streaming Gemini API response to standard ChatResponseUpdate objects.
        private IEnumerable<ChatResponseUpdate> ConvertGeminiResponseToChatResponseUpdates(GeminiResponse geminiResponse)
        {
            if (geminiResponse.Candidates != null)
            {
                foreach (var candidate in geminiResponse.Candidates)
                {
                    var contents = ConvertGeminiPartsToAIContent(candidate.Content?.Parts);
                    if (contents.Any())
                    {
                        yield return new ChatResponseUpdate
                        {
                            Role = ChatRole.Assistant,
                            Contents = contents,
                            ModelId = _modelName,
                            FinishReason = ConvertFinishReason(candidate.FinishReason, candidate.Content?.Parts),
                            RawRepresentation = geminiResponse
                        };
                    }
                }
            }

            if (geminiResponse.UsageMetadata != null)
            {
                var usageDetails = new UsageDetails
                {
                    InputTokenCount = geminiResponse.UsageMetadata.PromptTokenCount,
                    OutputTokenCount = geminiResponse.UsageMetadata.CandidatesTokenCount,
                    TotalTokenCount = geminiResponse.UsageMetadata.TotalTokenCount
                };
                yield return new ChatResponseUpdate { Contents = new List<AIContent> { new UsageContent(usageDetails) } };
            }
        }

        private static List<AIContent> ConvertGeminiPartsToAIContent(List<GeminiPart> parts)
        {
            var aiContents = new List<AIContent>();
            if (parts == null) return aiContents;

            foreach (var part in parts)
            {
                if (!string.IsNullOrEmpty(part.Text))
                {
                    aiContents.Add(new TextContent(part.Text));
                }
                else if (part.FunctionCall != null)
                {
                    aiContents.Add(new FunctionCallContent(
                        part.FunctionCall.Name,
                        part.FunctionCall.Name,
                        part.FunctionCall.Args));
                }
            }
            return aiContents;
        }

        private static List<GeminiPart> ConvertToGeminiParts(IEnumerable<AIContent> contents)
        {
            var parts = new List<GeminiPart>();
            if (contents == null) return parts;

            foreach (var content in contents)
            {
                switch (content)
                {
                    case TextContent textContent:
                        parts.Add(new GeminiPart { Text = textContent.Text });
                        break;
                    case DataContent dataContent:
                        parts.Add(new GeminiPart { InlineData = new GeminiInlineData { MimeType = dataContent.MediaType, Data = Convert.ToBase64String(dataContent.Data.Span) } });
                        break;
                    case UriContent uriContent:
                        parts.Add(new GeminiPart { FileData = new GeminiFileData { MimeType = uriContent.MediaType, FileUri = uriContent.Uri.ToString() } });
                        break;
                    case FunctionCallContent functionCall:
                        parts.Add(new GeminiPart { FunctionCall = new GeminiFunctionCall { Name = functionCall.Name, Args = functionCall.Arguments } });
                        break;
                    case FunctionResultContent functionResult:
                        parts.Add(new GeminiPart { FunctionResponse = new GeminiFunctionResponse { Name = functionResult.CallId, Response = new GeminiFunctionResponseData { Name = functionResult.CallId, Content = functionResult.Result } } });
                        break;
                }
            }
            return parts;
        }

        private static string ConvertRole(ChatRole role)
        {
            if (role == ChatRole.User) return "user";
            if (role == ChatRole.Assistant) return "model";
            if (role == ChatRole.Tool) return "user"; // Gemini use "user" role for function response.
            // The System role is handled by SystemInstruction and should not reach here.
            throw new ArgumentException($"Unsupported chat role: {role}", nameof(role));
        }

        private static GeminiFunctionDeclaration ConvertAIFunctionToGemini(AIFunction func)
        {
            var declaration = new GeminiFunctionDeclaration
            {
                Name = func.Name,
                Description = func.Description
            };

            var schemaJson = func.JsonSchema;
            if (schemaJson.ValueKind != JsonValueKind.Undefined && schemaJson.ValueKind != JsonValueKind.Null)
            {
                var schema = new GeminiSchema { Type = GeminiSchemaType.Object };
                if (schemaJson.TryGetProperty("properties", out var properties))
                {
                    schema.Properties = properties;
                }
                if (schemaJson.TryGetProperty("required", out var required))
                {
                    schema.Required = required.EnumerateArray().Select(x => x.GetString()).ToList();
                }
                declaration.Parameters = schema;
            }

            return declaration;
        }

        private static ChatFinishReason? ConvertFinishReason(string reason, List<GeminiPart> parts)
        {
            // Per Gemini API, if the model is asking for a tool call, the finishReason is "TOOL_CALLS".
            // We also check the content parts just in case, for robustness.
            if (reason == "TOOL_CALLS" || parts?.Any(p => p.FunctionCall != null) == true)
            {
                return ChatFinishReason.ToolCalls;
            }

            return reason switch
            {
                "STOP" => ChatFinishReason.Stop,
                "MAX_TOKENS" => ChatFinishReason.Length,
                "SAFETY" => ChatFinishReason.ContentFilter,
                "RECITATION" => ChatFinishReason.ContentFilter,
                _ => null,
            };
        }

        #region Nested Types for Gemini API Serialization

        private class GeminiRequest
        {
            [JsonPropertyName("contents")]
            public List<GeminiContent> Contents { get; set; }

            [JsonPropertyName("tools")]
            public List<GeminiTool> Tools { get; set; }

            [JsonPropertyName("tool_config")]
            public GeminiToolConfig ToolConfig { get; set; }

            [JsonPropertyName("system_instruction")]
            public GeminiSystemInstruction SystemInstruction { get; set; }

            [JsonPropertyName("generationConfig")]
            public GeminiGenerationConfig GenerationConfig { get; set; }
        }

        private class GeminiContent
        {
            [JsonPropertyName("role")]
            public string Role { get; set; }

            [JsonPropertyName("parts")]
            public List<GeminiPart> Parts { get; set; }
        }

        private class GeminiPart
        {
            [JsonPropertyName("text")]
            public string Text { get; set; }

            [JsonPropertyName("inline_data")]
            public GeminiInlineData InlineData { get; set; }

            [JsonPropertyName("file_data")]
            public GeminiFileData FileData { get; set; }

            [JsonPropertyName("functionCall")]
            public GeminiFunctionCall FunctionCall { get; set; }

            [JsonPropertyName("functionResponse")]
            public GeminiFunctionResponse FunctionResponse { get; set; }
        }

        private class GeminiInlineData
        {
            [JsonPropertyName("mime_type")]
            public string MimeType { get; set; }

            [JsonPropertyName("data")]
            public string Data { get; set; }
        }

        private class GeminiFileData
        {
            [JsonPropertyName("mime_type")]
            public string MimeType { get; set; }

            [JsonPropertyName("file_uri")]
            public string FileUri { get; set; }
        }

        private class GeminiFunctionCall
        {
            [JsonPropertyName("name")]
            public string Name { get; set; }

            [JsonPropertyName("args")]
            public IDictionary<string, object> Args { get; set; }
        }

        private class GeminiFunctionResponse
        {
            [JsonPropertyName("name")]
            public string Name { get; set; }

            [JsonPropertyName("response")]
            public GeminiFunctionResponseData Response { get; set; }
        }

        private class GeminiFunctionResponseData
        {
            [JsonPropertyName("name")]
            public string Name { get; set; }

            [JsonPropertyName("content")]
            public object Content { get; set; }
        }

        private class GeminiSystemInstruction
        {
            [JsonPropertyName("parts")]
            public List<GeminiPart> Parts { get; set; }
        }

        private class GeminiTool
        {
            [JsonPropertyName("function_declarations")]
            public List<GeminiFunctionDeclaration> FunctionDeclarations { get; set; }
        }

        private class GeminiToolConfig
        {
            [JsonPropertyName("function_calling_config")]
            public GeminiFunctionCallingConfig FunctionCallingConfig { get; set; }
        }

        private class GeminiFunctionCallingConfig
        {
            [JsonPropertyName("mode")]
            public GeminiFunctionCallingMode? Mode { get; set; }

            [JsonPropertyName("allowed_function_names")]
            public List<string> AllowedFunctionNames { get; set; }
        }

        private enum GeminiFunctionCallingMode { Auto, Any, None }

        private class GeminiFunctionDeclaration
        {
            [JsonPropertyName("name")]
            public string Name { get; set; }

            [JsonPropertyName("description")]
            public string Description { get; set; }

            [JsonPropertyName("parameters")]
            public GeminiSchema Parameters { get; set; }
        }

        private class GeminiSchema
        {
            [JsonPropertyName("type")]
            public GeminiSchemaType? Type { get; set; }

            [JsonPropertyName("properties")]
            public JsonElement? Properties { get; set; }

            [JsonPropertyName("required")]
            public List<string> Required { get; set; }

            [JsonPropertyName("description")]
            public string Description { get; set; }

            // *** MODIFICATION START ***
            // Added "items" property to handle schema for array elements
            [JsonPropertyName("items")]
            public GeminiSchema Items { get; set; }
            // *** MODIFICATION END ***
        }

        private enum GeminiSchemaType { String, Number, Integer, Boolean, Array, Object }

        private class GeminiGenerationConfig
        {
            [JsonPropertyName("temperature")]
            public float? Temperature { get; set; }

            [JsonPropertyName("topP")]
            public float? TopP { get; set; }

            [JsonPropertyName("topK")]
            public int? TopK { get; set; }

            [JsonPropertyName("maxOutputTokens")]
            public int? MaxOutputTokens { get; set; }

            [JsonPropertyName("stopSequences")]
            public List<string> StopSequences { get; set; }

            [JsonPropertyName("response_mime_type")]
            public string ResponseMimeType { get; set; }

            [JsonPropertyName("response_schema")]
            public GeminiSchema ResponseSchema { get; set; }

            [JsonPropertyName("thinkingConfig")]
            public GeminiThinkingConfig ThinkingConfig { get; set; }
        }

        private class GeminiThinkingConfig
        {
            [JsonPropertyName("thinkingBudget")]
            public int? ThinkingBudget { get; set; }

            [JsonPropertyName("includeThoughts")]
            public bool? IncludeThoughts { get; set; }
        }

        private class GeminiResponse
        {
            [JsonPropertyName("candidates")]
            public List<GeminiCandidate> Candidates { get; set; }

            [JsonPropertyName("usageMetadata")]
            public GeminiUsageMetadata UsageMetadata { get; set; }

            [JsonPropertyName("promptFeedback")]
            public GeminiPromptFeedback PromptFeedback { get; set; }
        }

        private class GeminiCandidate
        {
            [JsonPropertyName("content")]
            public GeminiContent Content { get; set; }

            [JsonPropertyName("finishReason")]
            public string FinishReason { get; set; }
        }

        private class GeminiUsageMetadata
        {
            [JsonPropertyName("promptTokenCount")]
            public long? PromptTokenCount { get; set; }

            [JsonPropertyName("candidatesTokenCount")]
            public long? CandidatesTokenCount { get; set; }

            [JsonPropertyName("totalTokenCount")]
            public long? TotalTokenCount { get; set; }
        }

        private class GeminiPromptFeedback
        {
            [JsonPropertyName("blockReason")]
            public string BlockReason { get; set; }
        }
        #endregion
    }
}
