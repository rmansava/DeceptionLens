using System.Net.Http.Headers;
using System.Text.Json;
using System.Globalization;
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

    public async Task<List<SearchResult>> ClipTextSearchAsync(string query, int topK = 50, string collection = "books")
    {
        try
        {
            if (collection == "all")
            {
                // Search all collections via POST endpoint
                var url = "/clip/text-all";
                _logger.LogInformation("CLIP text search ALL: {Query}, topK: {TopK}", query, topK);

                var requestBody = new { query = query, top_k = topK };
                var jsonContent = new StringContent(
                    JsonSerializer.Serialize(requestBody),
                    System.Text.Encoding.UTF8,
                    "application/json");

                var response = await _httpClient.PostAsync(url, jsonContent);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

                _logger.LogInformation("Found {Count} results from all collections", results.Count);
                return results;
            }
            else
            {
                var url = $"/clip/text?query={Uri.EscapeDataString(query)}&top_k={topK}&collection={Uri.EscapeDataString(collection)}";
                _logger.LogInformation("CLIP text search: {Query}, topK: {TopK}, collection: {Collection}", query, topK, collection);

                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var results = JsonSerializer.Deserialize<List<SearchResult>>(json) ?? new List<SearchResult>();

                _logger.LogInformation("Found {Count} results", results.Count);
                return results;
            }
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
        int topK = 50,
        string collection = "books")
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            string url;
            if (collection == "all")
            {
                url = $"/clip/search-all?top_k={topK}";
                _logger.LogInformation("CLIP image search ALL: {FileName}, topK: {TopK}", fileName, topK);
            }
            else
            {
                url = $"/clip/search?top_k={topK}&collection={Uri.EscapeDataString(collection)}&rerank=true";
                _logger.LogInformation("CLIP image search (rerank): {FileName}, topK: {TopK}, collection: {Collection}", fileName, topK, collection);
            }

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
        string collection = "images",
        double minScore = 0.0)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var minScoreText = minScore.ToString(CultureInfo.InvariantCulture);
            var url = $"/search/faces?top_k={topK}&collection={Uri.EscapeDataString(collection)}&min_score={Uri.EscapeDataString(minScoreText)}";

            _logger.LogInformation(
                "Face search: {FileName}, topK: {TopK}, collection: {Collection}, minScore: {MinScore}",
                fileName, topK, collection, minScore
            );

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

    public async Task<DiskSearchStartResponse?> DiskSearchAsync(
        Stream imageStream,
        string fileName,
        int topK = 50,
        string collection = "all")
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var streamContent = new StreamContent(imageStream);
            streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            content.Add(streamContent, "file", fileName);

            var url = $"/disk/search?top_k={topK}&rerank=true";
            if (!string.IsNullOrWhiteSpace(collection) && !string.Equals(collection, "all", StringComparison.OrdinalIgnoreCase))
            {
                url += $"&collections={Uri.EscapeDataString(collection)}";
            }

            _logger.LogInformation("DISK search queued: {FileName}, topK: {TopK}, collection: {Collection}", fileName, topK, collection);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var queued = JsonSerializer.Deserialize<DiskSearchStartResponse>(json);

            _logger.LogInformation("DISK search queued with search_id={SearchId}", queued?.SearchId);
            return queued;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "DISK search failed");
            throw;
        }
    }

    public async Task<DiskSearchStartResponse?> ResumeDiskSearchAsync(
        int sourceSearchId,
        int topK = 50,
        int k = 5,
        double threshold = 0.7)
    {
        try
        {
            var thresholdText = threshold.ToString(CultureInfo.InvariantCulture);
            var url = $"/disk/resume/{sourceSearchId}?top_k={topK}&k={k}&threshold={Uri.EscapeDataString(thresholdText)}&rerank=true";

            _logger.LogInformation(
                "Resuming DISK search from source {SourceSearchId}, topK: {TopK}, k: {K}, threshold: {Threshold}",
                sourceSearchId, topK, k, threshold
            );

            var response = await _httpClient.PostAsync(url, null);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var queued = JsonSerializer.Deserialize<DiskSearchStartResponse>(json);
            _logger.LogInformation("Resumed DISK search queued as search_id={SearchId}", queued?.SearchId);
            return queued;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to resume DISK search from {SourceSearchId}", sourceSearchId);
            throw;
        }
    }

    public async Task<SearchProgressResponse?> GetSearchProgressAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/search/progress");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<SearchProgressResponse>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get search progress");
            return null;
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

            content.Add(new StringContent(searchType), "search_type");
            content.Add(new StringContent(resultsJson), "results_json");
            if (!string.IsNullOrEmpty(queryText))
            {
                content.Add(new StringContent(queryText), "query_text");
            }
            if (durationMs.HasValue)
            {
                content.Add(new StringContent(durationMs.Value.ToString()), "search_duration_ms");
            }
            if (!string.IsNullOrEmpty(collection))
            {
                content.Add(new StringContent(collection), "collection");
            }

            _logger.LogInformation("Saving search history: type={SearchType}, results={Count}", searchType, results.Count);

            var response = await _httpClient.PostAsync("/history", content);
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

    public async Task<bool> StopSearchAsync(int searchId)
    {
        try
        {
            var response = await _httpClient.PostAsync($"/history/{searchId}/stop", null);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to stop search {SearchId}", searchId);
            return false;
        }
    }

    public async Task<List<ExclusionEntry>> GetExclusionsAsync(string searchType = "DISK")
    {
        try
        {
            var url = $"/exclusions?search_type={Uri.EscapeDataString(searchType)}";
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<List<ExclusionEntry>>(json) ?? new List<ExclusionEntry>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load exclusions");
            return new List<ExclusionEntry>();
        }
    }

    public async Task<bool> AddExclusionAsync(string path, string searchType = "DISK", string? reason = null)
    {
        try
        {
            var payload = new
            {
                path,
                search_type = searchType,
                reason
            };
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                System.Text.Encoding.UTF8,
                "application/json"
            );
            var response = await _httpClient.PostAsync("/exclusions", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to add exclusion for path {Path}", path);
            return false;
        }
    }

    public async Task<bool> RemoveExclusionAsync(string path, string searchType = "DISK")
    {
        try
        {
            var url = $"/exclusions?path={Uri.EscapeDataString(path)}&search_type={Uri.EscapeDataString(searchType)}";
            var response = await _httpClient.DeleteAsync(url);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to remove exclusion for path {Path}", path);
            return false;
        }
    }
}
