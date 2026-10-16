// onnxruntime-react-native 1.24.3 (android/build.gradle) compares React Native versions with org.gradle.util.VersionNumber,
// a class Gradle 9 no longer provides, so the library fails to configure under this project's Gradle 9.3.1 wrapper.
// Rewrite only that comparison as plain integer maths. Safe to run repeatedly; does nothing if the line is gone.
const fs = require('fs');
const path = require('path');
const file = path.join(__dirname, '..', 'node_modules', 'onnxruntime-react-native', 'android', 'build.gradle');
if (!fs.existsSync(file)) process.exit(0);
const text = fs.readFileSync(file, 'utf8');
const re = /VersionNumber\.parse\(REACT_NATIVE_VERSION\)\s*<\s*VersionNumber\.parse\("0\.71"\)/;
if (!re.test(text)) { console.log('[extcv] onnxruntime gradle: nothing to rewrite'); process.exit(0); }
fs.writeFileSync(file, text.replace(re, '(REACT_NATIVE_MINOR_VERSION < 71 && REACT_NATIVE_VERSION.startsWith("0."))'));
console.log('[extcv] onnxruntime gradle: VersionNumber comparison rewritten for Gradle 9');
