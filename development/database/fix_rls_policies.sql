-- RLS Policy Fix Script for Supabase
-- Run this in your Supabase SQL Editor to fix the drill_attempts RLS issue

-- First, let's see the current state
SELECT 'Current RLS status for drill_attempts:' as info;
SELECT tablename, rowsecurity FROM pg_tables WHERE tablename = 'drill_attempts';

SELECT 'Current policies for drill_attempts:' as info;
SELECT policyname, cmd, roles, qual, with_check FROM pg_policies WHERE tablename = 'drill_attempts';

-- Option 1: Allow anonymous users to insert drill attempts
-- This creates a policy that allows the 'anon' role to insert records

DROP POLICY IF EXISTS "Allow anonymous inserts on drill_attempts" ON drill_attempts;

CREATE POLICY "Allow anonymous inserts on drill_attempts"
ON drill_attempts
FOR INSERT
TO anon
WITH CHECK (true);

-- Option 2: Allow anonymous users to select their own drill attempts
-- (if you want them to be able to read back what they inserted)

DROP POLICY IF EXISTS "Allow anonymous selects on drill_attempts" ON drill_attempts;

CREATE POLICY "Allow anonymous selects on drill_attempts"
ON drill_attempts
FOR SELECT
TO anon
USING (true);

-- Option 3: If you have authenticated users, you might want user-specific policies
-- Uncomment these if you're using authentication:

/*
DROP POLICY IF EXISTS "Users can insert their own drill attempts" ON drill_attempts;

CREATE POLICY "Users can insert their own drill attempts"
ON drill_attempts
FOR INSERT
TO authenticated
WITH CHECK (auth.uid()::text = user_id);

DROP POLICY IF EXISTS "Users can view their own drill attempts" ON drill_attempts;

CREATE POLICY "Users can view their own drill attempts"
ON drill_attempts
FOR SELECT
TO authenticated
USING (auth.uid()::text = user_id);
*/

-- Check the same for user_progress table
SELECT 'Current RLS status for user_progress:' as info;
SELECT tablename, rowsecurity FROM pg_tables WHERE tablename = 'user_progress';

SELECT 'Current policies for user_progress:' as info;
SELECT policyname, cmd, roles, qual, with_check FROM pg_policies WHERE tablename = 'user_progress';

-- Allow anonymous access to user_progress if needed
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

-- Test the fix by attempting an insert
-- (Remove the ROLLBACK to make it permanent)
BEGIN;
INSERT INTO drill_attempts (user_id, drill_id, score, created_at) 
VALUES ('test-anon-user', 'test-drill', 95, NOW());
SELECT 'Test insert successful!' as result;
ROLLBACK;  -- Remove this line to keep the test record