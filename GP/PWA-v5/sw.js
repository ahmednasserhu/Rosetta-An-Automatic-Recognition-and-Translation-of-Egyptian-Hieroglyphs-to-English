const cacheName = "codeZone-v37"
const assets = [
  "/",
  "/PWA.html",
  "/css/PWA.css",
  "/js/app.js",
  "/imgaes/download.png",
  "/imgaes/download-2.png",
]
self.addEventListener("install", (installEvent) => {
  // console.log("installed", installEvent);
  installEvent.waitUntil(
    caches.open(cacheName).then((cache) => {
      cache.addAll(assets).then().catch()
    })
      .catch((err) => { })
  )

})

self.addEventListener("activate", (activateEvent) => {
  // console.log("activate sw.js", activateEvent);
  activateEvent.waituntil(
    caches.keys().then((keys) => {
      // console.log(keys);
      return Promise.all(
        keys.filter((key) => key != cacheName)
          .map((key) => caches.delete(key))
      )
    })
  )
})

self.addEventListener("fetch", (fetchEvent) => {
  // console.log("fetch", fetchEvent);
  fetchEvent.respondWith(
    caches.match(fetchEvent.request).then((res) => {
      return res || fetch(fetchEvent.request).then((fetchRes) => {
        return caches.open(cacheName).then((cache) => {
          cache.put(fetchEvent.request, fetchRes.clone())
          return fetchRes;
        })
      });
    })
  )
})
