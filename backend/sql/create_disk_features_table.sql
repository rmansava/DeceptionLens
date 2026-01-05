-- Create DiskFeatures table in trivia database
-- Stores pre-computed DISK keypoints and descriptors for LightGlue matching
-- This speeds up geometric verification from ~17 minutes to ~2-3 minutes per search

USE trivia;
GO

-- Drop if exists for clean recreation
IF OBJECT_ID('dbo.DiskFeatures', 'U') IS NOT NULL
    DROP TABLE dbo.DiskFeatures;
GO

CREATE TABLE dbo.DiskFeatures (
    Id INT IDENTITY(1,1) NOT NULL,

    -- Image identification
    ImagePath NVARCHAR(500) NOT NULL,
    BookName NVARCHAR(200) NULL,

    -- DISK features (compressed with gzip)
    -- Keypoints: (N, 2) array of float32 -> gzipped bytes
    -- Descriptors: (N, 128) array of float16 -> gzipped bytes
    Keypoints VARBINARY(MAX) NOT NULL,
    Descriptors VARBINARY(MAX) NOT NULL,

    -- Metadata for reconstruction
    KeypointCount SMALLINT NOT NULL,
    ImageWidth SMALLINT NOT NULL,
    ImageHeight SMALLINT NOT NULL,

    -- For padded dimensions (DISK requires multiples of 16)
    PaddedWidth SMALLINT NOT NULL,
    PaddedHeight SMALLINT NOT NULL,

    -- Timestamps
    CreatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
    UpdatedAt DATETIME2 NULL,

    -- Constraints
    CONSTRAINT PK_DiskFeatures PRIMARY KEY CLUSTERED (Id),
    CONSTRAINT UQ_DiskFeatures_ImagePath UNIQUE (ImagePath)
);
GO

-- Index for fast lookups by path (most common query)
CREATE NONCLUSTERED INDEX IX_DiskFeatures_ImagePath
ON dbo.DiskFeatures (ImagePath)
INCLUDE (KeypointCount, ImageWidth, ImageHeight, PaddedWidth, PaddedHeight);
GO

-- Index for book-based queries (for batch operations)
CREATE NONCLUSTERED INDEX IX_DiskFeatures_BookName
ON dbo.DiskFeatures (BookName)
INCLUDE (ImagePath, KeypointCount);
GO

-- Stats view
CREATE OR ALTER VIEW dbo.vw_DiskFeaturesStats AS
SELECT
    COUNT(*) AS TotalImages,
    COUNT(DISTINCT BookName) AS TotalBooks,
    SUM(CAST(KeypointCount AS BIGINT)) AS TotalKeypoints,
    AVG(KeypointCount) AS AvgKeypointsPerImage,
    SUM(DATALENGTH(Keypoints) + DATALENGTH(Descriptors)) / 1048576.0 AS TotalStorageMB,
    AVG(DATALENGTH(Keypoints) + DATALENGTH(Descriptors)) / 1024.0 AS AvgStoragePerImageKB,
    MIN(CreatedAt) AS FirstIndexed,
    MAX(CreatedAt) AS LastIndexed
FROM dbo.DiskFeatures;
GO

-- Stored procedure for bulk upsert (used by Python indexer)
CREATE OR ALTER PROCEDURE dbo.sp_UpsertDiskFeatures
    @ImagePath NVARCHAR(500),
    @BookName NVARCHAR(200),
    @Keypoints VARBINARY(MAX),
    @Descriptors VARBINARY(MAX),
    @KeypointCount SMALLINT,
    @ImageWidth SMALLINT,
    @ImageHeight SMALLINT,
    @PaddedWidth SMALLINT,
    @PaddedHeight SMALLINT
AS
BEGIN
    SET NOCOUNT ON;

    MERGE dbo.DiskFeatures AS target
    USING (SELECT @ImagePath AS ImagePath) AS source
    ON target.ImagePath = source.ImagePath
    WHEN MATCHED THEN
        UPDATE SET
            BookName = @BookName,
            Keypoints = @Keypoints,
            Descriptors = @Descriptors,
            KeypointCount = @KeypointCount,
            ImageWidth = @ImageWidth,
            ImageHeight = @ImageHeight,
            PaddedWidth = @PaddedWidth,
            PaddedHeight = @PaddedHeight,
            UpdatedAt = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN
        INSERT (ImagePath, BookName, Keypoints, Descriptors, KeypointCount,
                ImageWidth, ImageHeight, PaddedWidth, PaddedHeight)
        VALUES (@ImagePath, @BookName, @Keypoints, @Descriptors, @KeypointCount,
                @ImageWidth, @ImageHeight, @PaddedWidth, @PaddedHeight);
END;
GO

-- Stored procedure for bulk fetch (used during search)
CREATE OR ALTER PROCEDURE dbo.sp_GetDiskFeaturesBulk
    @ImagePaths NVARCHAR(MAX)  -- Comma-separated list of paths
AS
BEGIN
    SET NOCOUNT ON;

    -- Parse comma-separated paths and fetch features
    SELECT
        ImagePath,
        Keypoints,
        Descriptors,
        KeypointCount,
        ImageWidth,
        ImageHeight,
        PaddedWidth,
        PaddedHeight
    FROM dbo.DiskFeatures
    WHERE ImagePath IN (
        SELECT TRIM(value) FROM STRING_SPLIT(@ImagePaths, ',')
    );
END;
GO

PRINT 'DiskFeatures table and procedures created successfully';
PRINT 'Run: SELECT * FROM dbo.vw_DiskFeaturesStats to check status';
GO
