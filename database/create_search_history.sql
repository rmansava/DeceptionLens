-- Create image search history tables in the trivia database
-- Run this script against your MS SQL Server trivia database

USE trivia;
GO

-- ============================================================================
-- Image Search History Table
-- Stores metadata about each search performed
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ImageSearchHistory')
BEGIN
    CREATE TABLE ImageSearchHistory (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        SearchDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        SearchType NVARCHAR(50) NOT NULL,  -- 'DINOv2', 'CLIP', 'Deep Search', 'Face', 'Text'
        QueryText NVARCHAR(500) NULL,       -- For text searches
        QueryImage VARBINARY(MAX) NULL,     -- The uploaded image bytes
        QueryImageName NVARCHAR(255) NULL,  -- Original filename
        ResultCount INT NOT NULL DEFAULT 0,
        SearchDurationMs INT NULL,          -- How long the search took
        Collection NVARCHAR(100) NULL,      -- Which collection was searched
        Notes NVARCHAR(MAX) NULL            -- Optional user notes
    );

    CREATE INDEX IX_ImageSearchHistory_SearchDate ON ImageSearchHistory(SearchDate DESC);
    CREATE INDEX IX_ImageSearchHistory_SearchType ON ImageSearchHistory(SearchType);

    PRINT 'Created ImageSearchHistory table';
END
ELSE
    PRINT 'ImageSearchHistory table already exists';
GO

-- ============================================================================
-- Image Search Results Table
-- Stores individual results for each search
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ImageSearchResults')
BEGIN
    CREATE TABLE ImageSearchResults (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        SearchHistoryId INT NOT NULL,
        Rank INT NOT NULL,                  -- Position in results (1 = top result)
        ImagePath NVARCHAR(1000) NOT NULL,  -- Full path to the matched image
        Score FLOAT NOT NULL,               -- Similarity score
        VerifiedMatches INT NULL,           -- For geometric verification
        KeypointMatches INT NULL,           -- ORB keypoint matches
        TemplateScore FLOAT NULL,           -- Template matching score
        CombinedScore FLOAT NULL,           -- Combined re-ranking score

        CONSTRAINT FK_ImageSearchResults_ImageSearchHistory
            FOREIGN KEY (SearchHistoryId) REFERENCES ImageSearchHistory(Id)
            ON DELETE CASCADE
    );

    CREATE INDEX IX_ImageSearchResults_SearchHistoryId ON ImageSearchResults(SearchHistoryId);
    CREATE INDEX IX_ImageSearchResults_Rank ON ImageSearchResults(SearchHistoryId, Rank);

    PRINT 'Created ImageSearchResults table';
END
ELSE
    PRINT 'ImageSearchResults table already exists';
GO

-- ============================================================================
-- View for easy querying of search history with top result
-- ============================================================================
IF EXISTS (SELECT * FROM sys.views WHERE name = 'vw_ImageSearchHistoryWithTopResult')
    DROP VIEW vw_ImageSearchHistoryWithTopResult;
GO

CREATE VIEW vw_ImageSearchHistoryWithTopResult AS
SELECT
    h.Id,
    h.SearchDate,
    h.SearchType,
    h.QueryText,
    h.QueryImageName,
    h.ResultCount,
    h.SearchDurationMs,
    h.Collection,
    r.ImagePath AS TopResultPath,
    r.Score AS TopResultScore
FROM ImageSearchHistory h
LEFT JOIN ImageSearchResults r ON h.Id = r.SearchHistoryId AND r.Rank = 1;
GO

PRINT 'Created vw_ImageSearchHistoryWithTopResult view';
GO

-- ============================================================================
-- Stored Procedure to save a search with results
-- ============================================================================
IF EXISTS (SELECT * FROM sys.procedures WHERE name = 'sp_SaveImageSearchHistory')
    DROP PROCEDURE sp_SaveImageSearchHistory;
GO

CREATE PROCEDURE sp_SaveImageSearchHistory
    @SearchType NVARCHAR(50),
    @QueryText NVARCHAR(500) = NULL,
    @QueryImage VARBINARY(MAX) = NULL,
    @QueryImageName NVARCHAR(255) = NULL,
    @ResultCount INT,
    @SearchDurationMs INT = NULL,
    @Collection NVARCHAR(100) = NULL,
    @SearchHistoryId INT OUTPUT
AS
BEGIN
    SET NOCOUNT ON;

    INSERT INTO ImageSearchHistory (
        SearchType, QueryText, QueryImage, QueryImageName,
        ResultCount, SearchDurationMs, Collection
    )
    VALUES (
        @SearchType, @QueryText, @QueryImage, @QueryImageName,
        @ResultCount, @SearchDurationMs, @Collection
    );

    SET @SearchHistoryId = SCOPE_IDENTITY();
END
GO

PRINT 'Created sp_SaveImageSearchHistory procedure';
GO

-- ============================================================================
-- Stored Procedure to get search history with pagination
-- ============================================================================
IF EXISTS (SELECT * FROM sys.procedures WHERE name = 'sp_GetImageSearchHistory')
    DROP PROCEDURE sp_GetImageSearchHistory;
GO

CREATE PROCEDURE sp_GetImageSearchHistory
    @PageNumber INT = 1,
    @PageSize INT = 20,
    @SearchType NVARCHAR(50) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT
        h.Id,
        h.SearchDate,
        h.SearchType,
        h.QueryText,
        h.QueryImageName,
        h.ResultCount,
        h.SearchDurationMs,
        h.Collection,
        r.ImagePath AS TopResultPath,
        r.Score AS TopResultScore
    FROM ImageSearchHistory h
    LEFT JOIN ImageSearchResults r ON h.Id = r.SearchHistoryId AND r.Rank = 1
    WHERE (@SearchType IS NULL OR h.SearchType = @SearchType)
    ORDER BY h.SearchDate DESC
    OFFSET (@PageNumber - 1) * @PageSize ROWS
    FETCH NEXT @PageSize ROWS ONLY;
END
GO

PRINT 'Created sp_GetImageSearchHistory procedure';
GO

-- ============================================================================
-- Stored Procedure to get results for a specific search
-- ============================================================================
IF EXISTS (SELECT * FROM sys.procedures WHERE name = 'sp_GetImageSearchResults')
    DROP PROCEDURE sp_GetImageSearchResults;
GO

CREATE PROCEDURE sp_GetImageSearchResults
    @SearchHistoryId INT
AS
BEGIN
    SET NOCOUNT ON;

    -- Return search metadata
    SELECT
        Id, SearchDate, SearchType, QueryText, QueryImage,
        QueryImageName, ResultCount, SearchDurationMs, Collection, Notes
    FROM ImageSearchHistory
    WHERE Id = @SearchHistoryId;

    -- Return results
    SELECT
        Rank, ImagePath, Score, VerifiedMatches,
        KeypointMatches, TemplateScore, CombinedScore
    FROM ImageSearchResults
    WHERE SearchHistoryId = @SearchHistoryId
    ORDER BY Rank;
END
GO

PRINT 'Created sp_GetImageSearchResults procedure';
GO

PRINT '';
PRINT '============================================================';
PRINT 'Image search history tables created successfully!';
PRINT '============================================================';
GO
