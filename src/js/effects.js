import { cfgs } from './global.js';

class Particle {
    constructor(x, y, color, shouldConverge = false) {
        this.x = x;
        this.y = y;

        this.colorAlpha = 0;
        this.initialColor = color;
        this.targetColor = cfgs.foodColor;

        this.life = 1.0;
        this.size = Math.random() * 2 + 1;
        this.shouldConverge = shouldConverge;

        const angle = Math.random() * Math.PI * 2;
        const speed = 0.05 + Math.random() * 0.05;

        this.speedX = Math.cos(angle) * speed;
        this.speedY = Math.sin(angle) * speed;
        this.decay = 0.002 + Math.random() * 0.001;

        if (this.shouldConverge) {
            this.initialX = x;
            this.initialY = y;

            this.decay = 0;
            this.phaseTime = 0;
        }
    }

    lerpColor(color1, color2, alpha) {
        const rgb1 = color1.match(/\d+/g).map(Number);
        const rgb2 = color2.match(/\d+/g).map(Number);

        const r = Math.round(rgb1[0] + (rgb2[0] - rgb1[0]) * alpha);
        const g = Math.round(rgb1[1] + (rgb2[1] - rgb1[1]) * alpha);
        const b = Math.round(rgb1[2] + (rgb2[2] - rgb1[2]) * alpha);

        return `rgb(${r}, ${g}, ${b})`;
    }

    renderTick(deltaTime) {
        if (!this.shouldConverge || (this.phaseTime += deltaTime) < 500) {
            this.x += this.speedX * deltaTime;
            this.y += this.speedY * deltaTime;
            this.life -= this.decay * deltaTime;
            this.size *= Math.pow(0.999, deltaTime);
            this.speedX *= Math.pow(0.995, deltaTime);
            this.speedY *= Math.pow(0.995, deltaTime);
        } else {
            const dx = this.initialX - this.x;
            const dy = this.initialY - this.y;

            if (Math.sqrt(dx * dx + dy * dy) > 0.1) {
                this.x += dx * 0.005 * deltaTime;
                this.y += dy * 0.005 * deltaTime;
                this.colorAlpha = Math.min(1, this.colorAlpha + 0.002 * deltaTime);
            } else {
                this.life = 0;
            }
        }
    }

    render(ctx) {
        ctx.save();
        {
            ctx.translate(this.x, this.y);

            ctx.fillStyle = this.lerpColor(this.initialColor, this.targetColor, this.colorAlpha).replace('rgb', 'rgba').replace(')', `, ${Math.max(0, this.life)})`);

            ctx.beginPath();
            ctx.arc(0, 0, this.size, 0, Math.PI * 2);
            ctx.closePath();
            ctx.fill();
        }
        ctx.restore();
    }
}

export class EffectManager {
    constructor() {
        this.particles = [];
    }

    createFoodEffect(x, y) {
        for (let i = 0; i < 12; i++) {
            this.particles.push(new Particle(x, y, cfgs.foodColor));
        }
    }

    createSnakeEffect(x, y, color, shouldConverge = false) {
        for (let i = 0; i < 10; i++) {
            this.particles.push(new Particle(x, y, color, shouldConverge));
        }
    }

    renderTick(deltaTime) {
        this.particles.forEach(particle => particle.renderTick(deltaTime));
        this.particles = this.particles.filter(particle => particle.life > 0);
    }

    render(ctx) {
        this.particles.forEach(particle => particle.render(ctx));
    }
};