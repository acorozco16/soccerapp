# EAS Build Setup Guide

## Prerequisites
- [ ] Create Expo account at https://expo.dev
- [ ] Apple Developer Account (for iOS - $99/year) OR just build Android first
- [ ] Google Play Console Account (for Android - $25 one-time)

## Step 1: Login to EAS
```bash
cd "/Users/andreworozco/soccer app/soccer-training-app"
npx eas login
```
Enter your Expo account credentials when prompted.

## Step 2: Link Your Project
```bash
npx eas project:init
```
This will create a project on expo.dev and link it to your local app.

## Step 3: Build Development Versions

### For Android (Easier to Start)
```bash
npm run build:dev:android
# or
npx eas build --platform android --profile development
```

This will:
1. Build an APK file
2. Give you a download link when complete (15-30 mins)
3. You can install this APK directly on Android devices

### For iOS (Requires Apple Developer Account)
```bash
npm run build:dev:ios
# or
npx eas build --platform ios --profile development
```

If you don't have Apple certificates set up, EAS will guide you through:
1. Creating a development certificate
2. Setting up provisioning profiles
3. Building the app

## Step 4: Distribute to Testers

### Android Distribution
1. Download the APK from the build link
2. Options:
   - Send APK directly to testers
   - Upload to Google Play Console internal testing
   - Use Firebase App Distribution

### iOS Distribution
1. Use TestFlight (recommended)
2. EAS will help you upload to App Store Connect
3. Add testers via email in TestFlight

## Step 5: Running Development Builds
Once testers install the development build:
1. Start your local server: `npm start`
2. They can connect to your development server by:
   - Scanning QR code (if on same network)
   - Entering your server URL manually

## Troubleshooting

### "Not logged in" error
```bash
npx eas logout
npx eas login
```

### Build fails
- Check build logs at https://expo.dev
- Common issues:
  - Missing bundle identifier (we fixed this)
  - Package version conflicts
  - Asset issues

### Can't connect to development server
- Make sure your computer and phone are on same network
- Check firewall settings
- Try using tunnel: `npm start -- --tunnel`

## Next Steps
After successful builds:
1. Test on real devices
2. Add app icons and splash screen
3. Build preview/production versions when ready