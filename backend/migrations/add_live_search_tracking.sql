-- Migration: Add live search tracking to ImageSearchHistory
-- Run this on your SQL Server database before using live search features

-- Add Status column (in_progress, completed, failed)
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'ImageSearchHistory') AND name = 'Status')
BEGIN
    ALTER TABLE ImageSearchHistory
    ADD Status NVARCHAR(20) DEFAULT 'completed' NOT NULL;
END
GO

-- Add CurrentProgress column (e.g., "Searching chunk 5/441")
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'ImageSearchHistory') AND name = 'CurrentProgress')
BEGIN
    ALTER TABLE ImageSearchHistory
    ADD CurrentProgress NVARCHAR(100) NULL;
END
GO

-- Add TotalChunks column (for progress calculation)
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'ImageSearchHistory') AND name = 'TotalChunks')
BEGIN
    ALTER TABLE ImageSearchHistory
    ADD TotalChunks INT NULL;
END
GO

-- Update existing searches to have 'completed' status
UPDATE ImageSearchHistory
SET Status = 'completed'
WHERE Status IS NULL OR Status = '';
GO

PRINT 'Migration complete: Live search tracking columns added'
GO
