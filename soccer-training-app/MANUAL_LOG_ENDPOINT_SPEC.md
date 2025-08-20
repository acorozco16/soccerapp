# Manual Log Endpoint Specification

## Overview
This endpoint needs to be added to the DigitalOcean server at `https://soccertrainingapp.org` to handle manual practice logging from the Track flow.

## Endpoint Details

**URL:** `POST /drill/manual-log`
**Base URL:** `https://soccertrainingapp.org`
**Full URL:** `https://soccertrainingapp.org/drill/manual-log`

## Request Headers
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

## Request Body
```json
{
  "drill_type": "juggling",
  "count_detected": 25,
  "duration": 300,
  "manual_entry": true,
  "notes": "Practiced with weak foot",
  "confidence": 1.0,
  "touches_per_minute": "5.0",
  "juggle_type": "both_feet",
  "user_id": "user-uuid-from-jwt",
  "timestamp": "2025-08-19T03:45:00.000Z"
}
```

### Required Fields
- `drill_type`: string (e.g., "juggling")
- `count_detected`: integer (number of touches/repetitions)
- `duration`: integer (practice duration in seconds)
- `manual_entry`: boolean (always true for manual logs)
- `confidence`: float (always 1.0 for manual entries)
- `user_id`: string (extracted from JWT token)
- `timestamp`: string (ISO 8601 format)

### Optional Fields
- `notes`: string (user notes about the practice)
- `touches_per_minute`: string (calculated rate)
- `juggle_type`: string (only for juggling: "both_feet", "right_foot", "left_foot")

## Response

### Success (201 Created)
```json
{
  "success": true,
  "message": "Practice logged successfully",
  "id": "generated-log-id",
  "data": {
    "drill_type": "juggling",
    "count_detected": 25,
    "duration": 300,
    "timestamp": "2025-08-19T03:45:00.000Z"
  }
}
```

### Error (400 Bad Request)
```json
{
  "detail": "Invalid drill_type. Must be one of: juggling, dribbling, passing"
}
```

### Error (401 Unauthorized)
```json
{
  "detail": "Invalid or expired token"
}
```

## Database Storage
This should save to the same `drill_results` table used by video analysis, with:
- `manual_entry = true`
- `confidence = 1.0`
- All other fields as provided in request

## Implementation Notes
1. Validate JWT token and extract user_id
2. Validate drill_type against available drills
3. Ensure count_detected > 0
4. Ensure duration > 0
5. Save to drill_results table
6. Return success response with generated ID

## Current Status
The mobile app calls `drillService.logManualPractice()` which currently simulates success. Once this endpoint is implemented on the server, uncomment line 475 in `/src/services/drills.js`:

```javascript
// Change this:
// const response = await drillApi.post('/drill/manual-log', payload);

// To this:
const response = await drillApi.post('/drill/manual-log', payload);
```