import { useEffect, useRef } from "react";
import "../App.css";

export default function RymdBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const STAR_COLOR = "#fff";
    const STAR_SIZE = 3;
    const STAR_MIN_SCALE = 0.2;
    const OVERFLOW_THRESHOLD = 50;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    let scale = window.devicePixelRatio || 1;
    let width, height;

    const STAR_COUNT = (window.innerWidth + window.innerHeight) / 8;
    const stars = [];

    let pointerX = null;
    let pointerY = null;

    let velocity = { x: 0, y: 0, tx: 0, ty: 0, z: 0.0005 };
    let touchInput = false;

    function generate() {
      for (let i = 0; i < STAR_COUNT; i++) {
        stars.push({
          x: 0,
          y: 0,
          z: STAR_MIN_SCALE + Math.random() * (1 - STAR_MIN_SCALE),
        });
      }
    }

    function placeStar(star) {
      star.x = Math.random() * width;
      star.y = Math.random() * height;
    }

    function recycleStar(star) {
      let direction = "z";
      const vx = Math.abs(velocity.x);
      const vy = Math.abs(velocity.y);

      if (vx > 1 || vy > 1) {
        direction = vx > vy ? (velocity.x > 0 ? "l" : "r") : velocity.y > 0 ? "t" : "b";
      }

      star.z = STAR_MIN_SCALE + Math.random() * (1 - STAR_MIN_SCALE);

      if (direction === "l") {
        star.x = -OVERFLOW_THRESHOLD;
        star.y = Math.random() * height;
      } else if (direction === "r") {
        star.x = width + OVERFLOW_THRESHOLD;
        star.y = Math.random() * height;
      } else if (direction === "t") {
        star.x = Math.random() * width;
        star.y = -OVERFLOW_THRESHOLD;
      } else if (direction === "b") {
        star.x = Math.random() * width;
        star.y = height + OVERFLOW_THRESHOLD;
      } else {
        star.x = Math.random() * width;
        star.y = Math.random() * height;
      }
    }

    function resize() {
      scale = window.devicePixelRatio || 1;
      width = window.innerWidth * scale;
      height = window.innerHeight * scale;

      canvas.width = width;
      canvas.height = height;

      stars.forEach(placeStar);
    }

    function update() {
      velocity.tx *= 0.96;
      velocity.ty *= 0.96;

      velocity.x += (velocity.tx - velocity.x) * 0.8;
      velocity.y += (velocity.ty - velocity.y) * 0.8;

      stars.forEach((star) => {
        star.x += velocity.x * star.z;
        star.y += velocity.y * star.z;

        star.x += (star.x - width / 2) * velocity.z * star.z;
        star.y += (star.y - height / 2) * velocity.z * star.z;

        if (
          star.x < -OVERFLOW_THRESHOLD ||
          star.x > width + OVERFLOW_THRESHOLD ||
          star.y < -OVERFLOW_THRESHOLD ||
          star.y > height + OVERFLOW_THRESHOLD
        ) {
          recycleStar(star);
        }
      });
    }

    function render() {
      context.clearRect(0, 0, width, height);

      stars.forEach((star) => {
        context.beginPath();
        context.lineCap = "round";
        context.lineWidth = STAR_SIZE * star.z * scale;
        context.globalAlpha = 0.6;
        context.strokeStyle = STAR_COLOR;

        context.moveTo(star.x, star.y);
        context.lineTo(star.x + velocity.x * 2, star.y + velocity.y * 2);
        context.stroke();
      });
    }

    function step() {
      update();
      render();
      requestAnimationFrame(step);
    }

    function movePointer(x, y) {
      if (pointerX !== null && pointerY !== null) {
        velocity.tx += (x - pointerX) / (8 * scale);
        velocity.ty += (y - pointerY) / (8 * scale);
      }
      pointerX = x;
      pointerY = y;
    }

    const onMouseMove = (e) => {
      touchInput = false;
      movePointer(e.clientX, e.clientY);
    };

    const onTouchMove = (e) => {
      touchInput = true;
      movePointer(e.touches[0].clientX, e.touches[0].clientY);
      e.preventDefault();
    };

    const onLeave = () => {
      pointerX = null;
      pointerY = null;
    };

    generate();
    resize();
    step();

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("touchmove", onTouchMove);
    document.addEventListener("mouseleave", onLeave);

    return () => {
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("touchmove", onTouchMove);
      document.removeEventListener("mouseleave", onLeave);
    };
  }, []);

 return (
  <div className="rymd-wrapper">
    <canvas ref={canvasRef} id="rymd-bg" />
  </div>
);
}
