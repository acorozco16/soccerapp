# ⚽ Soccer Training Mobile App

> **Production Ready**: React Native + Expo mobile app for youth soccer practice tracking

## 🌐 **Current Status**
- **Production**: Live on TestFlight for iOS testing
- **Backend**: Connected to soccertrainingapp.org (DigitalOcean)
- **Features**: 8 drill types with AI analysis + manual logging
- **Distribution**: EAS Build system ready for App Store

## 🚀 Quick Start

### 1. Install Expo Go on Your Phone
- **iOS**: Search "Expo Go" in App Store
- **Android**: Search "Expo Go" in Google Play Store

### 2. Start the Development Server
```bash
npm start
```

### 3. Connect Your Phone
- Make sure your phone and computer are on the same WiFi network
- Open Expo Go app on your phone
- Scan the QR code shown in the terminal

### 4. Test the App
- The app will load on your phone
- Try the login screen (use any email/password for now)
- Changes you make will appear instantly!

## 📱 What's Working So Far

- ✅ Basic app structure
- ✅ Login screen UI
- ✅ Navigation setup
- ✅ API service ready
- ✅ Token storage ready

## 🔧 Development Tips

### For iOS Simulator (Mac only)
```bash
npm run ios
```

### For Android Emulator
```bash
npm run android
```

### To see console logs
```bash
npx expo start
```
Then press 'j' to open debugger

## ⚠️ Important Notes

1. **Backend Connection**: 
   - **Production**: Uses `https://soccertrainingapp.org` (current default)
   - **Development**: Change to `http://localhost:8000` in `src/constants/config.js`
   - **Local Testing**: Use your computer's IP address (e.g., `http://192.168.1.100:8000`)

2. **Authentication**: 
   - **Production**: Full Supabase authentication system
   - **Development**: Same auth system, works with local backend
   - Create new accounts directly through the mobile app

## 🎯 Next Steps

1. Add registration screen
2. Connect to real backend (change localhost to your IP)
3. Add drill selection screen
4. Add video recording
5. Show results

## 🐛 Troubleshooting

**"Metro bundler not found"**
- Run `npm start` again

**"Network request failed"**
- Make sure backend is running
- Change localhost to your computer's IP address

**App not updating**
- Shake phone and tap "Reload"
- Or press 'r' in terminal