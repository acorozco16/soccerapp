// Test script for drill server authentication
// Run with: node test-drill-server.js

const https = require('https');

// Test 1: Check if drill server is accessible
console.log('Testing drill server endpoints...\n');

// Test token exchange endpoint
const testTokenExchange = () => {
  const data = JSON.stringify({
    supabase_token: 'test-token'
  });

  const options = {
    hostname: 'soccertrainingapp.org',
    port: 443,
    path: '/auth/exchange',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  const req = https.request(options, (res) => {
    console.log(`Token Exchange Test - Status: ${res.statusCode}`);
    
    let responseBody = '';
    res.on('data', (chunk) => {
      responseBody += chunk;
    });
    
    res.on('end', () => {
      console.log(`Response: ${responseBody}\n`);
      testDrillAnalyze();
    });
  });

  req.on('error', (e) => {
    console.error(`Token exchange test error: ${e.message}\n`);
    testDrillAnalyze();
  });

  req.write(data);
  req.end();
};

// Test drill analyze endpoint
const testDrillAnalyze = () => {
  const options = {
    hostname: 'soccertrainingapp.org',
    port: 443,
    path: '/drill/analyze',
    method: 'POST',
    headers: {
      'Authorization': 'Bearer test-token'
    }
  };

  const req = https.request(options, (res) => {
    console.log(`Drill Analyze Test - Status: ${res.statusCode}`);
    
    let responseBody = '';
    res.on('data', (chunk) => {
      responseBody += chunk;
    });
    
    res.on('end', () => {
      console.log(`Response: ${responseBody}\n`);
      printRecommendations();
    });
  });

  req.on('error', (e) => {
    console.error(`Drill analyze test error: ${e.message}\n`);
    printRecommendations();
  });

  req.end();
};

const printRecommendations = () => {
  console.log('=== RECOMMENDATIONS ===');
  console.log('');
  console.log('1. If token exchange endpoint returns 404:');
  console.log('   → Implement /auth/exchange endpoint on your drill server');
  console.log('   → See DRILL_SERVER_AUTH.md for implementation details');
  console.log('');
  console.log('2. If drill analyze returns 401 with JWT error:');
  console.log('   → Configure drill server to accept Supabase JWT tokens');
  console.log('   → Use Supabase JWT secret for token verification');
  console.log('');
  console.log('3. Get your Supabase JWT secret from:');
  console.log('   → https://supabase.com/dashboard/project/nxumfeldylzpqwqlvszz');
  console.log('   → Settings → API → JWT Secret');
  console.log('');
};

// Start tests
testTokenExchange();