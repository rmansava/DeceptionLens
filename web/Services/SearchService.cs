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

    public async Task<byte[]?> GetVisualizationAsync(
        Stream queryImageStream,
        string fileName,
        string matchPath)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(queryImageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var url = $"/visualize?match_path={Uri.EscapeDataString(matchPath)}";

            _logger.LogInformation("Getting visualization for: {MatchPath}", matchPath);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            return await response.Content.ReadAsByteArrayAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Visualization failed for {MatchPath}", matchPath);
            return null;
        }
    }

    public async Task<List<SearchResult>> DeepSearchAsync(
        Stream imageStream,
        string fileName,
        int topK = 50,
        int retrievalK = 20000,
        int rerankK = 1000)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var url = $"/search/deep?top_k={topK}&retrieval_k={retrievalK}&rerank_k={rerankK}";

            _logger.LogInformation("Deep search: {FileName}, topK: {TopK}, retrievalK: {RetrievalK}, rerankK: {RerankK}",
                fileName, topK, retrievalK, rerankK);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

            _logger.LogInformation("Deep search found {Count} results", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Deep search failed");
            throw;
        }
    }

    // Search History Methods

    public async Task<SearchHistoryListResponse?> GetSearchHistoryAsync(int page = 1, int pageSize = 20, string? searchType = null)
    {
        try
        {
            var url = $"/history?page={page}&page_size={pageSize}";
            if (!string.IsNullOrEmpty(searchType))
            {
                url += $"&search_type={Uri.EscapeDataString(searchType)}";
            }

            _logger.LogInformation("Getting search history: page={Page}, pageSize={PageSize}", page, pageSize);

            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<SearchHistoryListResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get search history");
            return null;
        }
    }

    public async Task<SearchHistoryDetail?> GetSearchHistoryDetailAsync(int searchId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/history/{searchId}");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<SearchHistoryDetail>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get search history detail for {SearchId}", searchId);
            return null;
        }
    }

    public string GetSearchHistoryImageUrl(int searchId)
    {
        return $"{_apiBaseUrl}/history/{searchId}/image";
    }

    public async Task<SaveSearchResponse?> SaveSearchHistoryAsync(
        Stream? imageStream,
        string? fileName,
        string searchType,
        string? queryText,
        List<SearchResult> results,
        int? durationMs,
        string? collection)
    {
        try
        {
            // Convert results to JSON
            var resultsJson = JsonSerializer.Serialize(results.Select(r => new
            {
                path = r.Path,
                score = r.Score,
                verified_matches = r.VerifiedMatches
            }));

            using var content = new MultipartFormDataContent();

            // Add image if provided
            if (imageStream != null && !string.IsNullOrEmpty(fileName))
            {
                var streamContent = new StreamContent(imageStream);
                streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
                content.Add(streamContent, "file", fileName);
            }

            var url = $"/history?search_type={Uri.EscapeDataString(searchType)}&results_json={Uri.EscapeDataString(resultsJson)}";

            if (!string.IsNullOrEmpty(queryText))
                url += $"&query_text={Uri.EscapeDataString(queryText)}";
            if (durationMs.HasValue)
                url += $"&search_duration_ms={durationMs}";
            if (!string.IsNullOrEmpty(collection))
                url += $"&collection={Uri.EscapeDataString(collection)}";

            _logger.LogInformation("Saving search history: type={SearchType}, results={Count}", searchType, results.Count);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<SaveSearchResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save search history");
            return null;
        }
    }

    public async Task<bool> DeleteSearchHistoryAsync(int searchId)
    {
        try
        {
            var response = await _httpClient.DeleteAsync($"/history/{searchId}");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to delete search history {SearchId}", searchId);
            return false;
        }
    }
}
