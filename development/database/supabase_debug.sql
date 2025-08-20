-- Supabase Database Diagnostic Script
-- Run this in your Supabase SQL Editor to examine the drill_attempts table and RLS policies

-- 1. Check drill_attempts table structure
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default,
    character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'public' 
AND table_name = 'drill_attempts'
ORDER BY ordinal_position;

-- 2. Check if RLS is enabled on drill_attempts table
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables 
WHERE tablename = 'drill_attempts';

-- 3. Check current RLS policies on drill_attempts table
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies 
WHERE tablename = 'drill_attempts'
ORDER BY policyname;

-- 4. Check existing drill attempts (sample)
SELECT 
    id,
    user_id,
    drill_id,
    created_at,
    completed_at,
    score
FROM drill_attempts 
LIMIT 5;

-- 5. Count total drill attempts
SELECT COUNT(*) as total_drill_attempts FROM drill_attempts;

-- 6. Check user_progress table structure
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
AND table_name = 'user_progress'
ORDER BY ordinal_position;

-- 7. Check RLS policies on user_progress table
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies 
WHERE tablename = 'user_progress'
ORDER BY policyname;

-- 8. Check if there are any foreign key constraints
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
AND (tc.table_name = 'drill_attempts' OR tc.table_name = 'user_progress');

-- 9. Check current database user and role
SELECT current_user, current_role;

-- 10. Test INSERT permissions (this will help identify the specific RLS issue)
-- Note: This is a test query - you may want to rollback after testing
BEGIN;
INSERT INTO drill_attempts (user_id, drill_id, score, created_at) 
VALUES ('test-user-id', 'test-drill-id', 85, NOW());
ROLLBACK;