-- Image Hashes Table for Duplicate Detection
-- Used during indexing to skip already-indexed images

USE DeceptionLens;
GO

-- Drop existing table if it exists
IF OBJECT_ID('dbo.ImageHashes', 'U') IS NOT NULL
    DROP TABLE dbo.ImageHashes;
GO

CREATE TABLE dbo.ImageHashes (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    FileHash CHAR(64) NOT NULL,              -- SHA256 hash of file contents
    FilePath NVARCHAR(1000) NOT NULL,        -- Full path to the image
    Collection NVARCHAR(50) NOT NULL,        -- Collection name (books, print_ads, etc.)
    FileSize BIGINT NOT NULL,                -- File size in bytes
    IndexedDate DATETIME2 DEFAULT GETDATE(),

    -- Unique constraint on hash + collection (same image can be in multiple collections)
    CONSTRAINT UQ_ImageHashes_Hash_Collection UNIQUE (FileHash, Collection)
);
GO

-- Index for fast hash lookups
CREATE NONCLUSTERED INDEX IX_ImageHashes_FileHash
ON dbo.ImageHashes (FileHash);

-- Index for collection queries
CREATE NONCLUSTERED INDEX IX_ImageHashes_Collection
ON dbo.ImageHashes (Collection);

GO

-- Stored procedure to check if hash exists
CREATE OR ALTER PROCEDURE dbo.CheckImageHash
    @FileHash CHAR(64),
    @Collection NVARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;

    SELECT FilePath
    FROM dbo.ImageHashes
    WHERE FileHash = @FileHash AND Collection = @Collection;
END
GO

-- Stored procedure to add a new hash
CREATE OR ALTER PROCEDURE dbo.AddImageHash
    @FileHash CHAR(64),
    @FilePath NVARCHAR(1000),
    @Collection NVARCHAR(50),
    @FileSize BIGINT
AS
BEGIN
    SET NOCOUNT ON;

    -- Insert if not exists
    IF NOT EXISTS (SELECT 1 FROM dbo.ImageHashes WHERE FileHash = @FileHash AND Collection = @Collection)
    BEGIN
        INSERT INTO dbo.ImageHashes (FileHash, FilePath, Collection, FileSize)
        VALUES (@FileHash, @FilePath, @Collection, @FileSize);
    END
END
GO

-- Stored procedure to get collection stats
CREATE OR ALTER PROCEDURE dbo.GetImageHashStats
    @Collection NVARCHAR(50) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    IF @Collection IS NULL
    BEGIN
        SELECT
            Collection,
            COUNT(*) AS ImageCount,
            SUM(FileSize) / 1073741824.0 AS TotalSizeGB
        FROM dbo.ImageHashes
        GROUP BY Collection;
    END
    ELSE
    BEGIN
        SELECT
            Collection,
            COUNT(*) AS ImageCount,
            SUM(FileSize) / 1073741824.0 AS TotalSizeGB
        FROM dbo.ImageHashes
        WHERE Collection = @Collection
        GROUP BY Collection;
    END
END
GO

PRINT 'Image hashes table and procedures created successfully.';
