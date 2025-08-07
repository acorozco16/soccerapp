# 🔍 Supabase Authentication Debug Steps

## Issue Found
- Backend returns: "Database error saving new user"  
- Supabase Auth API returns: 500 Internal Server Error
- This means the issue is in Supabase project configuration, not our code

## Root Cause Analysis
The error occurs when Supabase tries to create a new user but fails to create the corresponding profile record. This typically happens due to:

1. **Missing database triggers** (most likely)
2. **Wrong profile table structure** 
3. **Email confirmation not disabled**
4. **Row Level Security blocking inserts**

## Fix Steps Required

### Step 1: Check Supabase Dashboard Settings
1. Go to: https://supabase.com/dashboard/project/nxumfeldylzpqwqlvszz
2. Navigate to: **Authentication > Settings**
3. **Disable email confirmation**:
   - Find "Enable email confirmations" 
   - Turn it **OFF** for development
   - Click **Save**

### Step 2: Fix Profile Table Structure
Current table probably has wrong columns. Need to check:
- What columns exist vs what our code expects
- Column names must match exactly

### Step 3: Create Database Trigger
Supabase needs a trigger to automatically create profile records when users sign up.

**SQL to run in Supabase SQL Editor:**
```sql
-- Check current table structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'profiles';

-- If table structure is wrong, recreate it
DROP TABLE IF EXISTS profiles;

CREATE TABLE profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email TEXT,
  full_name TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trigger to auto-create profiles
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create the trigger
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

### Step 4: Update Row Level Security
```sql
-- Enable RLS
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Allow users to read their own profile
CREATE POLICY "Users can view own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

-- Allow users to insert their own profile  
CREATE POLICY "Users can insert own profile" ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Allow users to update their own profile
CREATE POLICY "Users can update own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);
```

## Testing After Fix
1. Try registration from app again
2. Should work without "Database error saving new user"
3. User should be created in both auth.users and profiles tables

## Expected Result
- ✅ User created in Supabase Auth
- ✅ Profile automatically created in profiles table
- ✅ App login/registration works
- ✅ JWT tokens work correctly
- ✅ Ready for drill analysis with user tracking