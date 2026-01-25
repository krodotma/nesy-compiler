const staticPaths = new Set(["/._art_dept","/art_dept/._0fb64828691c74eb.css","/art_dept/._10099f28eda33175.css","/art_dept/._1e4e8c2645d1c2b3.css","/art_dept/._225cd3c56945aee5.css","/art_dept/._307d88c51eb5d2b5.css","/art_dept/._33448a8c072f922b.css","/art_dept/._561ec2e52a8b28a5.css","/art_dept/._66d3a781989942cf.css","/art_dept/._6f6b21f51a56d1e4.css","/art_dept/._79f8766297add503.css","/art_dept/._a4cc39aeb0f09297.css","/art_dept/._base_theme.css","/art_dept/._c5a5df7b26ab0f6e.css","/art_dept/._ce01a9642b1fdb12.css","/art_dept/._cebe0f7b44341c8e.css","/art_dept/._cyber_brutalist.css","/art_dept/._f01f23c2f571a567.css","/art_dept/._retro_terminal.css","/art_dept/._theme-minimal-1765988920.css","/art_dept/._theme-organic-1765988920.css","/art_dept/0fb64828691c74eb.css","/art_dept/10099f28eda33175.css","/art_dept/1e4e8c2645d1c2b3.css","/art_dept/225cd3c56945aee5.css","/art_dept/307d88c51eb5d2b5.css","/art_dept/33448a8c072f922b.css","/art_dept/561ec2e52a8b28a5.css","/art_dept/66d3a781989942cf.css","/art_dept/6f6b21f51a56d1e4.css","/art_dept/79f8766297add503.css","/art_dept/a4cc39aeb0f09297.css","/art_dept/base_theme.css","/art_dept/c5a5df7b26ab0f6e.css","/art_dept/ce01a9642b1fdb12.css","/art_dept/cebe0f7b44341c8e.css","/art_dept/cyber_brutalist.css","/art_dept/f01f23c2f571a567.css","/art_dept/retro_terminal.css","/art_dept/theme-minimal-1765988920.css","/art_dept/theme-organic-1765988920.css","/auralux-worklet.js","/favicon.ico","/favicon.svg","/icons/icon-192.png","/icons/icon-512.png","/manifest.json","/models/.cache/huggingface/.gitignore","/models/.cache/huggingface/download/files/silero_vad.onnx.lock","/models/.cache/huggingface/download/hubert-soft-0.95.onnx.lock","/models/.cache/huggingface/download/vocos_mel_24khz.onnx.lock","/models/download_models.sh","/models/hubert-soft-quantized.onnx","/models/silero_vad_v5.onnx","/models/vocos_q8.onnx","/q-manifest.json","/service-worker.js","/shader-test-clean.html","/shader-test.html","/sitemap.xml","/test_worley.html","/worklets/capture-worklet.js"]);
function isStaticPath(method, url) {
  if (method.toUpperCase() !== 'GET') {
    return false;
  }
  const p = url.pathname;
  if (p.startsWith("/build/")) {
    return true;
  }
  if (p.startsWith("/assets/")) {
    return true;
  }
  if (staticPaths.has(p)) {
    return true;
  }
  if (p.endsWith('/q-data.json')) {
    const pWithoutQdata = p.replace(/\/q-data.json$/, '');
    if (staticPaths.has(pWithoutQdata + '/')) {
      return true;
    }
    if (staticPaths.has(pWithoutQdata)) {
      return true;
    }
  }
  return false;
}
export { isStaticPath };