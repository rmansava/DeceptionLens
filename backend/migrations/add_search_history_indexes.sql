-- Migration: Add indexes for live search polling and result updates
-- Run this on your SQL Server database after live search tracking migration

-- Helps frequent history polling of in-progress sessions and latest-first history listing.
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_ImageSearchHistory_Status_SearchDate'
      AND object_id = OBJECT_ID(N'ImageSearchHistory')
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_ImageSearchHistory_Status_SearchDate
    ON ImageSearchHistory (Status, SearchDate DESC)
    INCLUDE (Id, SearchType, ResultCount, CurrentProgress, TotalChunks, SearchDurationMs, Collection);
END
GO

-- Speeds DELETE/SELECT by SearchHistoryId and ordered result retrieval by Rank.
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_ImageSearchResults_SearchHistoryId_Rank'
      AND object_id = OBJECT_ID(N'ImageSearchResults')
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_ImageSearchResults_SearchHistoryId_Rank
    ON ImageSearchResults (SearchHistoryId, Rank)
    INCLUDE (ImagePath, Score, VerifiedMatches, KeypointMatches, TemplateScore, CombinedScore);
END
GO

PRINT 'Migration complete: search history indexes added'
GO
