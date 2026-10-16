package com.muhammadfaizan.nutrisole

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider
import java.io.File
import java.security.MessageDigest

/**
 * NutriSole External-CV: ONNX Runtime opens models by file path, but the detectors ship as APK assets
 * (assets/models/<name>). ensure() copies an asset into app storage once (re-copies when the size differs)
 * and resolves { path, sha256, bytes, copied } so the app can show and check the model hash.
 */
class ExtCvModelFilesModule(context: ReactApplicationContext) : ReactContextBaseJavaModule(context) {
  override fun getName() = "ExtCvModelFiles"

  @ReactMethod
  fun ensure(name: String, expectedBytes: Double, promise: Promise) {
    Thread {
      try {
        require(!name.contains('/') && name.endsWith(".onnx")) { "bad model name" }
        val dir = File(reactApplicationContext.filesDir, "extcv-models").apply { mkdirs() }
        val out = File(dir, name)
        var copied = false
        if (!out.exists() || out.length() != expectedBytes.toLong()) {
          val tmp = File(dir, "$name.tmp")
          reactApplicationContext.assets.open("models/$name").use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
          if (!tmp.renameTo(out)) throw IllegalStateException("could not move model into place")
          copied = true
        }
        val digest = MessageDigest.getInstance("SHA-256")
        out.inputStream().use { input ->
          val buf = ByteArray(1 shl 20)
          while (true) { val n = input.read(buf); if (n < 0) break; digest.update(buf, 0, n) }
        }
        val map = Arguments.createMap()
        map.putString("path", out.absolutePath)
        map.putString("sha256", digest.digest().joinToString("") { "%02x".format(it) })
        map.putDouble("bytes", out.length().toDouble())
        map.putBoolean("copied", copied)
        promise.resolve(map)
      } catch (e: Exception) {
        promise.reject("E_EXTCV_MODEL", e.message ?: "model copy failed", e)
      }
    }.start()
  }
}

class ExtCvPackage : BaseReactPackage() {
  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    if (name == "ExtCvModelFiles") ExtCvModelFilesModule(reactContext) else null

  override fun getReactModuleInfoProvider() = ReactModuleInfoProvider {
    mapOf("ExtCvModelFiles" to ReactModuleInfo("ExtCvModelFiles", ExtCvModelFilesModule::class.java.name, false, false, false, false))
  }
}
