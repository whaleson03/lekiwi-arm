// LeKiwi Arm project page: back-to-top button, sections rising into view, the three synced videos and the
// embedded simulation (loaded on view, mouse wheel, full screen).
(function () {
  // ---------------------------------------------------------------- back to top
  const up = document.querySelector('.scroll-to-top');
  if (up) {
    up.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
    const covered = [document.getElementById('simEmbed'), document.querySelector('.sim-bar')].filter(Boolean);
    const overSim = () => {  // hidden while it would cover the simulation or its icons
      const b = up.getBoundingClientRect();
      return covered.some((el) => {
        const r = el.getBoundingClientRect();
        return r.top < b.bottom && r.bottom > b.top && r.left < b.right && r.right > b.left;
      });
    };
    const onScroll = () => up.classList.toggle('visible', window.scrollY > 300 && !overSim());
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();
  }

  // ---------------------------------------------------------------- sections rise in when they come into view
  const reveals = document.querySelectorAll('.reveal');
  if (reveals.length && 'IntersectionObserver' in window) {
    const ro = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add('in'); ro.unobserve(e.target); }
    }, { threshold: 0.12 });
    reveals.forEach((el) => ro.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add('in'));
  }

  // ---------------------------------------------------------------- three videos in sync
  // They start together and start over together once all have ended; each fades in with its first picture.
  // Click a video to pause / resume all; they also pause while scrolled out of view or in a hidden tab.
  const trio = document.getElementById('trio');
  const vids = trio ? Array.from(trio.querySelectorAll('video')) : [];
  const live = new Set(vids);  // the ones that loaded
  let inView = false, begun = false, paused = false;

  const shown = (v) => { v.classList.add('ready'); v.parentElement.classList.add('loaded'); };
  function playAll() {
    if (!inView || paused || document.hidden) return;
    const going = [...live].filter((v) => !v.ended);  // an ended one waits on its last frame
    const t = Math.min(...going.map((v) => v.currentTime));  // line them up first: they may have drifted apart
    for (const v of going) {
      if (Math.abs(v.currentTime - t) > 0.04) v.currentTime = t;
      v.play().catch(() => {});
    }
  }
  function restart() {
    for (const v of live) { v.pause(); v.currentTime = 0; }
    begun = true;
    playAll();
  }
  const ready = () => [...live].every((v) => v.readyState >= 3);
  function maybeBegin() {
    if (!begun && inView && ready()) restart();
  }
  for (const v of vids) {
    v.addEventListener('loadeddata', () => shown(v));
    v.addEventListener('canplay', maybeBegin);
    v.addEventListener('ended', () => { if ([...live].every((x) => x.ended)) restart(); });
    v.addEventListener('error', () => { live.delete(v); v.parentElement.classList.add('loaded'); maybeBegin(); });
    v.addEventListener('click', () => {
      paused = !paused;
      if (paused) for (const x of live) x.pause(); else playAll();
    });
    if (v.readyState >= 2) shown(v);
    if (v.error) v.dispatchEvent(new Event('error'));  // failed before this script ran
  }
  if (trio) {
    new IntersectionObserver((entries) => {
      inView = entries[0].isIntersecting;
      if (!inView) for (const v of live) v.pause();
      else if (begun) playAll();
      else maybeBegin();
    }, { threshold: 0.35 }).observe(trio);
    // a slow video (or iOS, which loads only on play()): show the posters and start anyway
    setTimeout(() => {
      for (const v of live) shown(v);
      if (!begun && inView) restart();
    }, 3000);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) for (const v of live) v.pause();
      else if (begun) playAll();
      else maybeBegin();
    });
  }

  // ---------------------------------------------------------------- the embedded simulation
  // loaded when it comes into view; the mouse wheel scrolls the page until the visitor clicks the 3D view
  const embed = document.getElementById('simEmbed'), frame = document.getElementById('simFrame');
  if (embed && frame) {
    if (frame.dataset.src) {
      new IntersectionObserver((entries, obs) => {
        if (!entries[0].isIntersecting) return;
        frame.src = frame.dataset.src;
        obs.disconnect();
      }).observe(embed);
    }
    let engaged = false;
    frame.addEventListener('mouseleave', () => { engaged = false; });
    frame.addEventListener('load', () => {
      let w;
      try { w = frame.contentWindow; if (!w.document) return; } catch (e) { return; }  // not same-origin
      w.addEventListener('pointerdown', () => { engaged = true; }, true);
      w.addEventListener('wheel', (e) => {
        if (engaged || document.fullscreenElement) return;
        const t = e.target;
        if (!t || !t.closest || !t.closest('#view')) return;  // over the panel: it scrolls as usual
        e.preventDefault();
        e.stopImmediatePropagation();
        const k = e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? window.innerHeight : 1;
        try { window.scrollBy({ top: e.deltaY * k, behavior: 'instant' }); } catch (err) { window.scrollBy(0, e.deltaY * k); }
      }, { capture: true, passive: false });
    });
    const full = document.getElementById('simFull');
    if (full) {
      if (!embed.requestFullscreen) full.hidden = true;
      full.addEventListener('click', (e) => {
        e.preventDefault();
        if (embed.requestFullscreen) embed.requestFullscreen().then(() => frame.focus()).catch(() => {});
      });
    }
  }
})();
