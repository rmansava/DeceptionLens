using DinoDeceptionLens.Web.Models;

namespace DinoDeceptionLens.Web.Services;

public interface ISearchService
{
    Task<List<SearchResult>> SearchAsync(Stream imageStream, string fileName, int topK = 50, string collection = "images", bool verify = false);
    Task<StatsResponse?> GetStatsAsync(string collection = "images");
    Task<HealthResponse?> GetHealthAsync();
    Task<List<string>> GetCollectionsAsync();
    string GetImageUrl(string path);

    // CLIP search methods
    Task<List<SearchResult>> ClipTextSearchAsync(string query, int topK = 50);
    Task<List<SearchResult>> ClipImageSearchAsync(Stream imageStream, string fileName, int topK = 50);
    Task<ClipStatsResponse?> GetClipStatsAsync();

    // Face search methods
    Task<List<SearchResult>> FaceSearchAsync(Stream imageStream, string fileName, int topK = 50, string collection = "images");
}
