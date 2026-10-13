/* PLATE — behaviour for the static sites. No dependencies, ~3 KB.
   Everything here is an enhancement: with JS off the page is complete. */
(() => {
  const doc = document.documentElement
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  const store = {
    get(k) { try { return localStorage.getItem(k) } catch { return null } },
    set(k, v) { try { localStorage.setItem(k, v) } catch { /* private mode */ } },
  }

  /* Theme: system by default, a toggle remembers an explicit choice. */
  const saved = store.get('plate-theme')
  if (saved === 'light' || saved === 'dark') doc.dataset.theme = saved
  document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const dark = doc.dataset.theme
        ? doc.dataset.theme === 'dark'
        : !matchMedia('(prefers-color-scheme: light)').matches
      doc.dataset.theme = dark ? 'light' : 'dark'
      store.set('plate-theme', doc.dataset.theme)
    })
  })

  /* Mobile menu. */
  document.querySelectorAll('.bar__menu').forEach((btn) => {
    const nav = document.getElementById(btn.getAttribute('aria-controls'))
    if (!nav) return
    btn.addEventListener('click', () => {
      const open = nav.classList.toggle('is-open')
      btn.setAttribute('aria-expanded', String(open))
    })
    nav.addEventListener('click', (e) => {
      if (e.target.closest('a')) { nav.classList.remove('is-open'); btn.setAttribute('aria-expanded', 'false') }
    })
  })

  /* Reels: real footage plays only while on screen, never with reduced motion. */
  const fmt = (t) => {
    const s = Math.floor(t), cs = Math.floor((t - s) * 100)
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}.${String(cs).padStart(2, '0')}`
  }
  const reels = [...document.querySelectorAll('.reel video')]
  reels.forEach((v) => {
    const reel = v.closest('.reel')
    const tc = reel.querySelector('.reel__tc')
    const btn = reel.querySelector('.reel__toggle')
    let wanted = !reduce
    const sync = () => {
      reel.classList.toggle('is-playing', !v.paused)
      if (!btn) return
      btn.setAttribute('aria-label', v.paused ? 'Play clip' : 'Pause clip')
      btn.innerHTML = v.paused ? '&#9654;' : '&#10073;&#10073;'
    }
    v.muted = true
    v.addEventListener('play', sync)
    v.addEventListener('pause', sync)
    if (tc) v.addEventListener('timeupdate', () => { tc.textContent = fmt(v.currentTime) })
    if (btn) btn.addEventListener('click', () => { wanted = v.paused; v.paused ? v.play().catch(() => {}) : v.pause() })
    v._plate = () => wanted
    sync()
  })
  if ('IntersectionObserver' in window && reels.length) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const v = e.target
        if (e.isIntersecting && v._plate()) {
          if (v.preload === 'none') v.preload = 'auto'
          v.play().catch(() => {})
        } else if (!e.isIntersecting) v.pause()
      }
    }, { threshold: 0.35 })
    reels.forEach((v) => io.observe(v))
  }

  /* Terminal: lines already in the HTML replay once when scrolled into view. */
  if (!reduce && 'IntersectionObserver' in window) {
    const terms = document.querySelectorAll('.term[data-replay]')
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue
        io.unobserve(e.target)
        const lines = [...e.target.querySelectorAll('.ln')]
        lines.forEach((l) => (l.style.visibility = 'hidden'))
        lines.forEach((l, i) => setTimeout(() => (l.style.visibility = ''), 260 + i * 320))
      }
    }, { threshold: 0.5 })
    terms.forEach((t) => io.observe(t))
  }

  /* Crosshair: a drafting readout that follows the pointer over the hero. */
  if (!reduce && matchMedia('(hover: hover) and (pointer: fine)').matches) {
    document.querySelectorAll('.hero[data-xhair]').forEach((hero) => {
      const x = document.createElement('div')
      x.className = 'xhair'
      x.setAttribute('aria-hidden', 'true')
      x.innerHTML = '<i class="xh"></i><i class="xv"></i><span class="xr"></span>'
      hero.prepend(x)
      const [h, v, r] = x.children
      let raf = 0
      hero.addEventListener('pointermove', (e) => {
        cancelAnimationFrame(raf)
        raf = requestAnimationFrame(() => {
          const b = hero.getBoundingClientRect()
          const px = e.clientX - b.left, py = e.clientY - b.top
          h.style.top = `${py}px`; v.style.left = `${px}px`
          r.style.left = `${px + 10}px`; r.style.top = `${py + 8}px`
          r.textContent = `X ${String(Math.round(px)).padStart(4, '0')}  Y ${String(Math.round(py)).padStart(4, '0')}`
        })
      })
    })
  }
})()
