using System.Text.Json.Serialization;

namespace DinoDeceptionLens.Web.Models;

public class SearchResult
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("score")]
    public double Score { get; set; }

    [JsonPropertyName("verified_matches")]
    public int VerifiedMatches { get; set; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, object>? Metadata { get; set; }

    public string FileName => System.IO.Path.GetFileName(Path);
}

public class StatsResponse
{
    [JsonPropertyName("visual_count")]
    public int VisualCount { get; set; }

    [JsonPropertyName("face_count")]
    public int FaceCount { get; set; }
}

public class HealthResponse
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("searcher_loaded")]
    public bool SearcherLoaded { get; set; }

    [JsonPropertyName("db_path")]
    public string DbPath { get; set; } = string.Empty;
}

public class CollectionsResponse
{
    [JsonPropertyName("collections")]
    public List<string> Collections { get; set; } = new();
}

public class ClipStatsResponse
{
    [JsonPropertyName("total_images")]
    public int TotalImages { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("index_path")]
    public string IndexPath { get; set; } = string.Empty;
}

// Search History Models

public class SearchHistoryEntry
{
    [JsonPropertyName("Id")]
    public int Id { get; set; }

    [JsonPropertyName("SearchDate")]
    public string SearchDate { get; set; } = string.Empty;

    [JsonPropertyName("SearchType")]
    public string SearchType { get; set; } = string.Empty;

    [JsonPropertyName("QueryText")]
    public string? QueryText { get; set; }

    [JsonPropertyName("QueryImageName")]
    public string? QueryImageName { get; set; }

    [JsonPropertyName("ResultCount")]
    public int ResultCount { get; set; }

    [JsonPropertyName("SearchDurationMs")]
    public int? SearchDurationMs { get; set; }

    [JsonPropertyName("Collection")]
    public string? Collection { get; set; }

    [JsonPropertyName("TopResultPath")]
    public string? TopResultPath { get; set; }

    [JsonPropertyName("TopResultScore")]
    public double? TopResultScore { get; set; }

    [JsonPropertyName("Status")]
    public string? Status { get; set; }

    [JsonPropertyName("CurrentProgress")]
    public string? CurrentProgress { get; set; }

    [JsonPropertyName("TotalChunks")]
    public int? TotalChunks { get; set; }

    [JsonPropertyName("TopResultVotes")]
    public int? TopResultVotes { get; set; }

    [JsonPropertyName("SecondResultVotes")]
    public int? SecondResultVotes { get; set; }

    public bool IsInProgress => Status == "in_progress";

    public bool HasPossibleHit
    {
        get
        {
            if (!SearchType.Contains("DISK", StringComparison.OrdinalIgnoreCase))
                return false;

            if (!TopResultVotes.HasValue || !SecondResultVotes.HasValue)
                return false;

            var top = TopResultVotes.Value;
            var second = Math.Max(SecondResultVotes.Value, 1);

            if (top < 120)
                return false;

            // Avoid early noise for long-running searches.
            if (IsInProgress && ProgressPercent < 10)
                return false;

            var ratio = (double)top / second;
            var margin = top - second;
            return ratio >= 1.5 && margin >= 50;
        }
    }

    public double ProgressPercent
    {
        get
        {
            if (TotalChunks is null or 0) return 0;
            if (CurrentProgress != null && CurrentProgress.Contains('/'))
            {
                var parts = CurrentProgress.Split(' ').LastOrDefault()?.Split('/');
                if (parts?.Length == 2 && int.TryParse(parts[0], out var current))
                    return (double)current / TotalChunks.Value * 100;
            }
            return 0;
        }
    }

    public DateTime ParsedDate => DateTime.TryParse(SearchDate, out var dt) ? dt : DateTime.MinValue;
}

public class SearchHistoryListResponse
{
    [JsonPropertyName("entries")]
    public List<SearchHistoryEntry> Entries { get; set; } = new();

    [JsonPropertyName("total")]
    public int Total { get; set; }

    [JsonPropertyName("page")]
    public int Page { get; set; }

    [JsonPropertyName("page_size")]
    public int PageSize { get; set; }
}

public class SearchHistoryDetailResult
{
    [JsonPropertyName("Rank")]
    public int Rank { get; set; }

    [JsonPropertyName("ImagePath")]
    public string ImagePath { get; set; } = string.Empty;

    [JsonPropertyName("Score")]
    public double Score { get; set; }

    [JsonPropertyName("VerifiedMatches")]
    public int? VerifiedMatches { get; set; }

    [JsonPropertyName("KeypointMatches")]
    public int? KeypointMatches { get; set; }

    [JsonPropertyName("TemplateScore")]
    public double? TemplateScore { get; set; }

    [JsonPropertyName("CombinedScore")]
    public double? CombinedScore { get; set; }

    [JsonPropertyName("MatchX1")]
    public double? MatchX1 { get; set; }

    [JsonPropertyName("MatchY1")]
    public double? MatchY1 { get; set; }

    [JsonPropertyName("MatchX2")]
    public double? MatchX2 { get; set; }

    [JsonPropertyName("MatchY2")]
    public double? MatchY2 { get; set; }

    [JsonPropertyName("MatchInliers")]
    public int? MatchInliers { get; set; }

    [JsonPropertyName("MatchTotal")]
    public int? MatchTotal { get; set; }

    public bool HasHighlight =>
        MatchX1.HasValue && MatchY1.HasValue &&
        MatchX2.HasValue && MatchY2.HasValue &&
        MatchX2.Value > MatchX1.Value &&
        MatchY2.Value > MatchY1.Value;

    public string FileName => System.IO.Path.GetFileName(ImagePath);
}

public class SearchHistoryDetail
{
    [JsonPropertyName("Id")]
    public int Id { get; set; }

    [JsonPropertyName("SearchDate")]
    public string SearchDate { get; set; } = string.Empty;

    [JsonPropertyName("SearchType")]
    public string SearchType { get; set; } = string.Empty;

    [JsonPropertyName("QueryText")]
    public string? QueryText { get; set; }

    [JsonPropertyName("QueryImageName")]
    public string? QueryImageName { get; set; }

    [JsonPropertyName("ResultCount")]
    public int ResultCount { get; set; }

    [JsonPropertyName("SearchDurationMs")]
    public int? SearchDurationMs { get; set; }

    [JsonPropertyName("Collection")]
    public string? Collection { get; set; }

    [JsonPropertyName("Notes")]
    public string? Notes { get; set; }

    [JsonPropertyName("Status")]
    public string? Status { get; set; }

    [JsonPropertyName("CurrentProgress")]
    public string? CurrentProgress { get; set; }

    [JsonPropertyName("TotalChunks")]
    public int? TotalChunks { get; set; }

    public bool IsInProgress => Status == "in_progress";

    [JsonPropertyName("Results")]
    public List<SearchHistoryDetailResult> Results { get; set; } = new();
}

public class SaveSearchResponse
{
    [JsonPropertyName("id")]
    public int Id { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;
}

public class SearchProgressResponse
{
    [JsonPropertyName("stage")]
    public string Stage { get; set; } = "idle";

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("current")]
    public int Current { get; set; }

    [JsonPropertyName("total")]
    public int Total { get; set; }

    [JsonPropertyName("cache_hits")]
    public int CacheHits { get; set; }

    [JsonPropertyName("cache_misses")]
    public int CacheMisses { get; set; }

    [JsonPropertyName("rate")]
    public double Rate { get; set; }

    [JsonPropertyName("eta_seconds")]
    public int EtaSeconds { get; set; }
}

public class DiskSearchStartResponse
{
    [JsonPropertyName("search_id")]
    public int SearchId { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("queue_position")]
    public int QueuePosition { get; set; }

    [JsonPropertyName("total_chunks")]
    public int TotalChunks { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;
}

public class ExclusionEntry
{
    [JsonPropertyName("Id")]
    public int Id { get; set; }

    [JsonPropertyName("Path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("PathNormalized")]
    public string PathNormalized { get; set; } = string.Empty;

    [JsonPropertyName("SearchType")]
    public string SearchType { get; set; } = string.Empty;

    [JsonPropertyName("Reason")]
    public string? Reason { get; set; }

    [JsonPropertyName("CreatedDate")]
    public string CreatedDate { get; set; } = string.Empty;
}
