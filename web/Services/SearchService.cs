using System.Net.Http.Headers;
using System.Text.Json;
using DinoDeceptionLens.Web.Models;

namespace DinoDeceptionLens.Web.Services;

public class SearchService : ISearchService
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<SearchService> _logger;
    private readonly string _apiBaseUrl;

    public SearchService(HttpClient httpClient, ILogger<SearchService> logger, IConfiguration configuration)
    {
        _httpClient = httpClient;
        _logger = logger;
        _apiBaseUrl = configuration.GetValue<string>("ApiUrl") ?? "http://localhost:8000";

        // Ensure BaseAddress is set (fallback if not configured via DI)
        if (_httpClient.BaseAddress == null)
        {
            _httpClient.BaseAddress = new Uri(_apiBaseUrl);
        }
    }

    public async Task<List<SearchResult>> SearchAsync(
        Stream imageStream,
        string fileName,
        int topK = 50,
        string collection = "images",
        bool verify = false)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var url = $"/search?top_k={topK}&collection={Uri.EscapeDataString(collection)}&verify={verify.ToString().ToLower()}";

            _logger.LogInformation("Searching with file: {FileName}, topK: {TopK}, collection: {Collection}", fileName, topK, collection);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

            _logger.LogInformation("Found {Count} results", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Search failed");
            throw;
        }
    }

    public async Task<StatsResponse?> GetStatsAsync(string collection = "images")
    {
        try
        {
            var response = await _httpClient.GetAsync($"/stats?collection={Uri.EscapeDataString(collection)}");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<StatsResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get stats");
            return null;
        }
    }

    public async Task<HealthResponse?> GetHealthAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/health");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<HealthResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get health");
            return null;
        }
    }

    public async Task<List<string>> GetCollectionsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/collections");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<CollectionsResponse>(json);
            return result?.Collections ?? new List<string>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get collections");
            return new List<string>();
        }
    }

    public string GetImageUrl(string path)
    {
        return $"{_apiBaseUrl}/image?path={Uri.EscapeDataString(path)}";
    }

    public async Task<List<SearchResult>> ClipTextSearchAsync(string query, int topK = 50)
    {
        try
        {
            var url = $"/clip/text?query={Uri.EscapeDataString(query)}&top_k={topK}";

            _logger.LogInformation("CLIP text search: {Query}, topK: {TopK}", query, topK);

            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

            _logger.LogInformation("Found {Count} results", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "CLIP text search failed");
            throw;
        }
    }

    public async Task<List<SearchResult>> ClipImageSearchAsync(
        Stream imageStream,
        string fileName,
        int topK = 50)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var url = $"/clip/search?top_k={topK}";

            _logger.LogInformation("CLIP image search: {FileName}, topK: {TopK}", fileName, topK);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

            _logger.LogInformation("Found {Count} results", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "CLIP image search failed");
            throw;
        }
    }

    public async Task<ClipStatsResponse?> GetClipStatsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/clip/stats");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<ClipStatsResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get CLIP stats");
            return null;
        }
    }

    public async Task<List<SearchResult>> FaceSearchAsync(
        Stream imageStream,
        string fileName,
        int topK = 50,
        string collection = "images")
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var url = $"/search/faces?top_k={topK}&collection={Uri.EscapeDataString(collection)}";

            _logger.LogInformation("Face search: {FileName}, topK: {TopK}, collection: {Collection}", fileName, topK, collection);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

            _logger.LogInformation("Found {Count} face results", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Face search failed");
            throw;
        }
    }
}
