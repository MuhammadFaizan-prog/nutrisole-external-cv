// react-native-worklets' Babel plugin must be listed last; it compiles the camera frame-processor worklet in extcv/.
module.exports = { presets: ['module:@react-native/babel-preset'], plugins: ['react-native-worklets/plugin'] };
