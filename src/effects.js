class Particle {
    constructor(x, y, color, type, shouldConverge = false) {
        this.x = x;
        this.y = y;
        this.initialColor = color || 'rgb(156, 39, 176)';
        this.targetColor = 'rgb(156, 39, 176)';
        this.currentColor = this.initialColor;
        this.type = type;
        this.size = Math.random() * 2 + 1;
        this.life = 1.0;

        const angle = Math.random() * Math.PI * 2;

        if (this.type === 'food') {
            const speed = 0.05 + Math.random() * 0.05;

            this.speedX = Math.cos(angle) * speed;
            this.speedY = Math.sin(angle) * speed;
            this.decay = 0.002 + Math.random() * 0.001;
            this.food = new Food((x - gridSize / 2) / gridSize, (y - gridSize / 2) / gridSize);
        } else {
            const speed = 0.05 + Math.random() * 0.05;

            this.shouldConverge = shouldConverge;
            this.initialX = x;
            this.initialY = y;

            this.speedX = Math.cos(angle) * speed;
            this.speedY = Math.sin(angle) * speed;
            this.explosionPhase = true;
            this.phaseTime = 0;
            this.colorTransitionProgress = 0;

            this.decay = (this.shouldConverge ? 0.33 : 1) * (0.002 + Math.random() * 0.001);
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
        if (this.type === 'food') {
            this.x += this.speedX * deltaTime;
            this.y += this.speedY * deltaTime;
            this.life -= this.decay * deltaTime;
            this.size *= Math.pow(0.999, deltaTime);
            this.speedX *= Math.pow(0.995, deltaTime);
            this.speedY *= Math.pow(0.995, deltaTime);
            this.food.size *= Math.pow(0.995, deltaTime);
        } else {
            if (this.shouldConverge) {
                this.phaseTime += deltaTime;
                
                if (this.explosionPhase) {
                    this.x += this.speedX * deltaTime;
                    this.y += this.speedY * deltaTime;
                    this.size *= Math.pow(0.999, deltaTime);
                    this.speedX *= Math.pow(0.995, deltaTime);
                    this.speedY *= Math.pow(0.995, deltaTime);
                    
                    if (this.phaseTime > 500) {
                        this.explosionPhase = false;
                    }
                } else {
                    const dx = this.initialX - this.x;
                    const dy = this.initialY - this.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist > 0.1) {
                        this.x += dx * 0.005 * deltaTime;
                        this.y += dy * 0.005 * deltaTime;

                        this.colorTransitionProgress = Math.min(1, this.colorTransitionProgress + 0.002 * deltaTime);
                        this.currentColor = this.lerpColor(this.initialColor, this.targetColor, this.colorTransitionProgress);
                    } else {
                        this.life = 0;
                    }
                }
            } else {
                this.x += this.speedX * deltaTime;
                this.y += this.speedY * deltaTime;
                this.life -= this.decay * deltaTime;
                this.size *= Math.pow(0.999, deltaTime);
                this.speedX *= Math.pow(0.995, deltaTime);
                this.speedY *= Math.pow(0.995, deltaTime);
            }
        }
    }

    render(ctx) {
        if (this.food) {
            this.food.render(ctx);
        }

        let colorStr = this.type === 'snake' && this.shouldConverge ? this.currentColor : this.initialColor;

        ctx.save();
        ctx.translate(this.x, this.y);

        ctx.fillStyle = colorStr.replace('rgb', 'rgba').replace(')', `, ${Math.max(0, this.life)})`);
        ctx.beginPath();
        ctx.arc(0, 0, this.size, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }
}

class EffectManager {
    constructor() {
        this.particles = [];
    }

    createFoodEffect(x, y) {
        for (let i = 0; i < 12; i++) {
            this.particles.push(new Particle(x, y, 'rgb(156, 39, 176)', 'food'));
        }
    }

    createSnakeEffect(x, y, color, shouldConverge = false) {
        for (let i = 0; i < 10; i++) {
            this.particles.push(new Particle(x, y, color, 'snake', shouldConverge));
        }
    }

    renderTick(deltaTime) {
        this.particles.forEach(particle => particle.renderTick(deltaTime));
        this.particles = this.particles.filter(particle => particle.life > 0);
    }

    render(ctx) {
        this.particles.forEach(particle => particle.render(ctx));
    }
}