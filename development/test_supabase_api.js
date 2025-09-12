// Supabase API Test Script
// Run this with Node.js to test your Supabase connection and identify RLS issues

const SUPABASE_URL = 'https://nxumfeldylzpqwqlvszz.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im54dW1mZWxkeWx6cHF3cWx2c3p6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM5MTY1NDksImV4cCI6MjA2OTQ5MjU0OX0.D2WvA9Ld2YalWbum6qi5CBvXxmj75v1BuDb-NKrJkxo';

async function testSupabaseConnection() {
    console.log('Testing Supabase connection and RLS policies...\n');

    // Test 1: Basic connection test
    console.log('1. Testing basic connection...');
    try {
        const response = await fetch(`${SUPABASE_URL}/rest/v1/`, {
            headers: {
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        console.log(`Connection status: ${response.status}`);
        if (response.status === 200) {
            console.log('✓ Basic connection successful\n');
        } else {
            console.log('✗ Connection failed\n');
            return;
        }
    } catch (error) {
        console.log('✗ Connection error:', error.message, '\n');
        return;
    }

    // Test 2: Try to read from drill_attempts table
    console.log('2. Testing read access to drill_attempts table...');
    try {
        const response = await fetch(`${SUPABASE_URL}/rest/v1/drill_attempts?select=*&limit=1`, {
            headers: {
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        
        console.log(`Read status: ${response.status}`);
        if (response.status === 200) {
            const data = await response.json();
            console.log('✓ Read access successful');
            console.log('Sample data:', JSON.stringify(data, null, 2));
        } else {
            const errorData = await response.text();
            console.log('✗ Read access failed');
            console.log('Error:', errorData);
        }
    } catch (error) {
        console.log('✗ Read error:', error.message);
    }
    console.log('');

    // Test 3: Try to insert into drill_attempts table
    console.log('3. Testing insert access to drill_attempts table...');
    const testRecord = {
        user_id: 'test-user-' + Date.now(),
        drill_id: 'test-drill-' + Date.now(),
        score: 85,
        created_at: new Date().toISOString()
    };

    try {
        const response = await fetch(`${SUPABASE_URL}/rest/v1/drill_attempts`, {
            method: 'POST',
            headers: {
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            body: JSON.stringify(testRecord)
        });
        
        console.log(`Insert status: ${response.status}`);
        if (response.status === 201) {
            const data = await response.json();
            console.log('✓ Insert access successful');
            console.log('Inserted record:', JSON.stringify(data, null, 2));
        } else {
            const errorData = await response.text();
            console.log('✗ Insert access failed');
            console.log('Error:', errorData);
            
            // Parse the error to understand RLS issue
            try {
                const errorJson = JSON.parse(errorData);
                if (errorJson.message && errorJson.message.includes('row-level security policy')) {
                    console.log('\n🔍 RLS POLICY VIOLATION DETECTED:');
                    console.log('This confirms the RLS policy is blocking anonymous inserts.');
                    console.log('You need to either:');
                    console.log('1. Create an RLS policy that allows anonymous inserts');
                    console.log('2. Use authenticated requests with proper user context');
                    console.log('3. Temporarily disable RLS for testing (not recommended for production)');
                }
            } catch (parseError) {
                console.log('Raw error response:', errorData);
            }
        }
    } catch (error) {
        console.log('✗ Insert error:', error.message);
    }
    console.log('');

    // Test 4: Check user_progress table
    console.log('4. Testing read access to user_progress table...');
    try {
        const response = await fetch(`${SUPABASE_URL}/rest/v1/user_progress?select=*&limit=1`, {
            headers: {
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        
        console.log(`User progress read status: ${response.status}`);
        if (response.status === 200) {
            const data = await response.json();
            console.log('✓ User progress read access successful');
            console.log('Sample data:', JSON.stringify(data, null, 2));
        } else {
            const errorData = await response.text();
            console.log('✗ User progress read access failed');
            console.log('Error:', errorData);
        }
    } catch (error) {
        console.log('✗ User progress read error:', error.message);
    }
}

// Run the tests
if (typeof require !== 'undefined' && require.main === module) {
    testSupabaseConnection();
}

// Export for use in other scripts
if (typeof module !== 'undefined') {
    module.exports = { testSupabaseConnection };
}