/**
 * Starfield & Cosmic Background Engine
 * Deep black galaxy backdrop with subtle low-opacity nebula textures and soft twinkling stars.
 */

export function initStarfield(canvasId = 'starfield-canvas') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let animationFrameId;
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  const starCount = Math.floor((width * height) / 3500);
  const stars = [];

  for (let i = 0; i < starCount; i++) {
    stars.push({
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 1.4 + 0.2,
      baseAlpha: Math.random() * 0.5 + 0.15,
      alpha: Math.random() * 0.5 + 0.15,
      twinkleSpeed: (Math.random() * 0.015 + 0.003) * (Math.random() > 0.5 ? 1 : -1),
      speedY: Math.random() * 0.05 + 0.01,
      color: getRandomStarColor()
    });
  }

  function getRandomStarColor() {
    const colors = [
      'rgba(240, 240, 245, ', // Silver-white
      'rgba(215, 215, 230, ', // Soft silver
      'rgba(190, 200, 220, ', // Faint silver-blue
      'rgba(220, 210, 235, '  // Faint silver-purple
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }

  let shootingStar = null;
  function spawnShootingStar() {
    if (Math.random() < 0.008 && !shootingStar) {
      shootingStar = {
        x: Math.random() * width * 0.8,
        y: Math.random() * height * 0.4,
        length: Math.random() * 70 + 40,
        speed: Math.random() * 8 + 10,
        angle: Math.PI / 4 + (Math.random() - 0.5) * 0.2,
        opacity: 0.6,
        life: 0,
        maxLife: 35
      };
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    // True Black Galaxy Base
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Faint subtle violet-blue nebula ambient patch 1
    const nebula1 = ctx.createRadialGradient(
      width * 0.75, height * 0.25, 40,
      width * 0.75, height * 0.25, width * 0.5
    );
    nebula1.addColorStop(0, 'rgba(80, 70, 120, 0.025)');
    nebula1.addColorStop(0.5, 'rgba(40, 50, 90, 0.012)');
    nebula1.addColorStop(1, 'transparent');
    ctx.fillStyle = nebula1;
    ctx.fillRect(0, 0, width, height);

    // Faint subtle blue-silver nebula ambient patch 2
    const nebula2 = ctx.createRadialGradient(
      width * 0.25, height * 0.7, 60,
      width * 0.25, height * 0.7, width * 0.5
    );
    nebula2.addColorStop(0, 'rgba(50, 80, 110, 0.02)');
    nebula2.addColorStop(0.6, 'rgba(30, 45, 75, 0.008)');
    nebula2.addColorStop(1, 'transparent');
    ctx.fillStyle = nebula2;
    ctx.fillRect(0, 0, width, height);

    // Draw Stars
    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];

      s.alpha += s.twinkleSpeed;
      if (s.alpha > 0.85 || s.alpha < 0.1) {
        s.twinkleSpeed = -s.twinkleSpeed;
      }

      s.y -= s.speedY;
      if (s.y < 0) {
        s.y = height;
        s.x = Math.random() * width;
      }

      ctx.fillStyle = s.color + Math.max(0, Math.min(1, s.alpha)) + ')';
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
      ctx.fill();
    }

    // Shooting Star
    spawnShootingStar();
    if (shootingStar) {
      shootingStar.life++;
      const currentOpacity = (1 - shootingStar.life / shootingStar.maxLife) * 0.6;
      const endX = shootingStar.x - Math.cos(shootingStar.angle) * shootingStar.length;
      const endY = shootingStar.y - Math.sin(shootingStar.angle) * shootingStar.length;

      const shootGrad = ctx.createLinearGradient(
        shootingStar.x, shootingStar.y,
        endX, endY
      );
      shootGrad.addColorStop(0, `rgba(240, 240, 255, ${currentOpacity})`);
      shootGrad.addColorStop(0.4, `rgba(180, 190, 220, ${currentOpacity * 0.5})`);
      shootGrad.addColorStop(1, 'transparent');

      ctx.strokeStyle = shootGrad;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(shootingStar.x, shootingStar.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      shootingStar.x += Math.cos(shootingStar.angle) * shootingStar.speed;
      shootingStar.y += Math.sin(shootingStar.angle) * shootingStar.speed;

      if (shootingStar.life >= shootingStar.maxLife) {
        shootingStar = null;
      }
    }

    animationFrameId = requestAnimationFrame(draw);
  }

  function handleResize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }

  window.addEventListener('resize', handleResize);
  draw();

  return () => {
    cancelAnimationFrame(animationFrameId);
    window.removeEventListener('resize', handleResize);
  };
}
