-- Corrected RLS Policy Fix Script for Supabase
-- Run this in your Supabase SQL Editor to fix the drill_attempts RLS issue
-- This script uses the correct column structure based on backend code analysis

-- First, let's see the current state
SELECT 'Current RLS status for drill_attempts:' as info;
SELECT tablename, rowsecurity FROM pg_tables WHERE tablename = 'drill_attempts';

SELECT 'Current policies for drill_attempts:' as info;
SELECT policyname, cmd, roles, qual, with_check FROM pg_policies WHERE tablename = 'drill_attempts';

-- SOLUTION: Allow anonymous users to insert drill attempts
-- This is necessary because the backend uses the anonymous key to insert results

DROP POLICY IF EXISTS "Allow anonymous inserts on drill_attempts" ON drill_attempts;

CREATE POLICY "Allow anonymous inserts on drill_attempts"
ON drill_attempts
FOR INSERT
TO anon
WITH CHECK (true);

-- Allow anonymous users to select drill attempts (for progress tracking)
DROP POLICY IF EXISTS "Allow anonymous selects on drill_attempts" ON drill_attempts;

CREATE POLICY "Allow anonymous selects on drill_attempts"
ON drill_attempts
FOR SELECT
TO anon
USING (true);

-- Check the same for user_progress table
SELECT 'Current RLS status for user_progress:' as info;
SELECT tablename, rowsecurity FROM pg_tables WHERE tablename = 'user_progress';

SELECT 'Current policies for user_progress:' as info;
SELECT policyname, cmd, roles, qual, with_check FROM pg_policies WHERE tablename = 'user_progress';

-- Allow anonymous access to user_progress for backend operations
DROP POLICY IF EXISTS "Allow anonymous inserts on user_progress" ON user_progress;

CREATE POLICY "Allow anonymous inserts on user_progress"
ON user_progress
FOR INSERT
TO anon
WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anonymous selects on user_progress" ON user_progress;

CREATE POLICY "Allow anonymous selects on user_progress"
ON user_progress
FOR SELECT
TO anon
USING (true);

DROP POLICY IF EXISTS "Allow anonymous updates on user_progress" ON user_progress;

CREATE POLICY "Allow anonymous updates on user_progress"
ON user_progress
FOR UPDATE
TO anon
USING (true)
WITH CHECK (true);

-- Display final status
SELECT 'Final policies for drill_attempts:' as info;
SELECT policyname, cmd, roles, qual, with_check FROM pg_policies WHERE tablename = 'drill_attempts';

SELECT 'Final policies for user_progress:' as info;
SELECT policyname, cmd, roles, qual, with_check FROM pg_policies WHERE tablename = 'user_progress';

-- Test the fix by attempting an insert with correct structure
BEGIN;
INSERT INTO drill_attempts (user_id, drill_type, results, video_filename, created_at) 
VALUES (
    'test-anon-user', 
    'juggling', 
    '{"count_detected": 4, "duration": 11.6, "benchmark_met": false}'::jsonb,
    'test_video.mp4',
    NOW()
);
SELECT 'Test insert successful!' as result;
ROLLBACK;  -- Remove this line to keep the test record

-- Additional check: Verify the backend can now save drill attempts
SELECT 'RLS policies successfully updated for anonymous backend access' as final_status;