# Soccer Training App

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

1. **Backend Connection**: Currently points to `localhost:8000`. You'll need to:
   - Start your backend server: `cd ../backend && python3 main.py`
   - For phone testing, change `localhost` to your computer's IP address in `src/constants/config.js`

2. **Test Credentials**: Since the backend requires real authentication, you'll need to:
   - Register a new account through the app, or
   - Use existing credentials if you have them

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