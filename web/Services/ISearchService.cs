using DinoDeceptionLens.Web.Models;

namespace DinoDeceptionLens.Web.Services;

public interface ISearchService
{
    Task<List<SearchResult>> SearchAsync(Stream imageStream, string fileName, int topK = 50, string collection = "images", bool verify = false);
    Task<StatsResponse?> GetStatsAsync(string collection = "images");
    Task<HealthResponse?> GetHealthAsync();
    Task<List<string>> GetCollectionsAsync();
    string GetImageUrl(string path);

    // CLIP search methods (collection = "all" searches all collections)
    Task<List<SearchResult>> ClipTextSearchAsync(string query, int topK = 50, string collection = "books");
    Task<List<SearchResult>> ClipImageSearchAsync(Stream imageStream, string fileName, int topK = 50, string collection = "books");
    Task<ClipStatsResponse?> GetClipStatsAsync();

    // Face search methods
    Task<List<SearchResult>> FaceSearchAsync(Stream imageStream, string fileName, int topK = 50, string collection = "images");

    // DISK keypoint search (queued job with history polling)
    Task<DiskSearchStartResponse?> DiskSearchAsync(Stream imageStream, string fileName, int topK = 50, string collection = "all");

    // Visualization methods
    Task<byte[]?> GetVisualizationAsync(Stream queryImageStream, string fileName, string matchPath);

    // Deep search (parallel CLIP + DINOv2 with reranking)
    Task<List<SearchResult>> DeepSearchAsync(Stream imageStream, string fileName, int topK = 50, int retrievalK = 20000, int rerankK = 1000);

    // Progress tracking for long-running searches
    Task<SearchProgressResponse?> GetSearchProgressAsync();

    // Search history methods
    Task<SearchHistoryListResponse?> GetSearchHistoryAsync(int page = 1, int pageSize = 20, string? searchType = null);
    Task<SearchHistoryDetail?> GetSearchHistoryDetailAsync(int searchId);
    string GetSearchHistoryImageUrl(int searchId);
    Task<SaveSearchResponse?> SaveSearchHistoryAsync(Stream? imageStream, string? fileName, string searchType, string? queryText, List<SearchResult> results, int? durationMs, string? collection);
    Task<bool> DeleteSearchHistoryAsync(int searchId);
    Task<bool> StopSearchAsync(int searchId);
}
