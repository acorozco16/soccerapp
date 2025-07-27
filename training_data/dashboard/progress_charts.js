/**
 * Progress Charts for Training Dashboard
 * Handles chart visualization and updates
 */

class TrainingCharts {
    constructor() {
        this.performanceChart = null;
        this.initCharts();
    }

    initCharts() {
        this.initPerformanceChart();
    }

    initPerformanceChart() {
        const canvas = document.getElementById('performanceChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Simple canvas-based chart implementation
        this.performanceChart = {
            canvas: canvas,
            ctx: ctx,
            data: [],
            maxDataPoints: 50
        };

        this.drawPerformanceChart();
    }

    updatePerformanceChart(data) {
        if (!this.performanceChart) return;

        // Update data
        this.performanceChart.data = data.slice(-this.performanceChart.maxDataPoints);
        this.drawPerformanceChart();
    }

    drawPerformanceChart() {
        const chart = this.performanceChart;
        const ctx = chart.ctx;
        const canvas = chart.canvas;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!chart.data || chart.data.length === 0) {
            this.drawNoDataMessage(ctx, canvas);
            return;
        }

        // Chart dimensions
        const padding = 60;
        const chartWidth = canvas.width - 2 * padding;
        const chartHeight = canvas.height - 2 * padding;

        // Draw axes
        this.drawAxes(ctx, padding, chartWidth, chartHeight);

        // Draw data lines
        this.drawAccuracyLine(ctx, chart.data, padding, chartWidth, chartHeight);
        this.drawLegend(ctx, canvas.width, padding);
    }

    drawAxes(ctx, padding, width, height) {
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, padding + height);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding + height);
        ctx.lineTo(padding + width, padding + height);
        ctx.stroke();

        // Y-axis labels (0-100%)
        ctx.fillStyle = '#666';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        
        for (let i = 0; i <= 10; i++) {
            const y = padding + height - (i / 10) * height;
            const value = i * 10;
            ctx.fillText(`${value}%`, padding - 10, y + 4);
            
            // Grid lines
            if (i > 0) {
                ctx.strokeStyle = '#f0f0f0';
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(padding + width, y);
                ctx.stroke();
            }
        }

        // X-axis labels
        ctx.textAlign = 'center';
        const data = this.performanceChart.data;
        if (data.length > 0) {
            const step = Math.max(1, Math.floor(data.length / 8));
            for (let i = 0; i < data.length; i += step) {
                const x = padding + (i / (data.length - 1)) * width;
                const timestamp = new Date(data[i].timestamp);
                const label = timestamp.toLocaleDateString();
                ctx.fillText(label, x, padding + height + 20);
            }
        }
    }

    drawAccuracyLine(ctx, data, padding, width, height) {
        if (data.length < 2) return;

        ctx.strokeStyle = '#1e3c72';
        ctx.lineWidth = 2;
        ctx.beginPath();

        data.forEach((point, index) => {
            const x = padding + (index / (data.length - 1)) * width;
            const accuracy = point.accuracy || 0;
            const y = padding + height - (accuracy / 100) * height;

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw data points
        ctx.fillStyle = '#1e3c72';
        data.forEach((point, index) => {
            const x = padding + (index / (data.length - 1)) * width;
            const accuracy = point.accuracy || 0;
            const y = padding + height - (accuracy / 100) * height;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    drawLegend(ctx, canvasWidth, padding) {
        ctx.fillStyle = '#1e3c72';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Model Accuracy Over Time', padding, 30);
    }

    drawNoDataMessage(ctx, canvas) {
        ctx.fillStyle = '#999';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No training data available', canvas.width / 2, canvas.height / 2);
    }
}

// Real-time data simulation for demo purposes
class DemoDataGenerator {
    constructor() {
        this.isGenerating = false;
        this.demoData = {
            system: {
                health_status: 'healthy',
                cpu_usage: 45,
                memory_usage: 60,
                disk_usage: 35,
                active_processes: 3
            },
            collection: {
                overall_progress: 0.7,
                total_items_collected: 3500,
                active_sources: 2
            },
            training: {
                status: 'running',
                current_stage: 'Stage 2',
                progress: 0.65,
                current_accuracy: 0.84,
                best_accuracy: 0.89
            },
            alerts: [
                {
                    type: 'warning',
                    component: 'system',
                    message: 'Disk usage approaching 80%'
                }
            ],
            recent_activities: [
                {
                    component: 'training',
                    message: 'Model training started for Stage 2',
                    timestamp: new Date().toISOString(),
                    status: 'running'
                },
                {
                    component: 'collection',
                    message: 'YouTube scraping completed: 500 videos',
                    timestamp: new Date(Date.now() - 300000).toISOString(),
                    status: 'completed'
                },
                {
                    component: 'deployment',
                    message: 'Model deployed to staging',
                    timestamp: new Date(Date.now() - 600000).toISOString(),
                    status: 'completed'
                }
            ],
            performance_history: this.generatePerformanceHistory()
        };
    }

    generatePerformanceHistory() {
        const history = [];
        const baseAccuracy = 70;
        const now = new Date();

        for (let i = 30; i >= 0; i--) {
            const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
            const accuracy = baseAccuracy + Math.random() * 20 + (30 - i) * 0.5;
            
            history.push({
                timestamp: date.toISOString(),
                accuracy: Math.min(95, accuracy),
                model_version: `v1.${30 - i}`
            });
        }

        return history;
    }

    startDemo() {
        if (this.isGenerating) return;
        
        this.isGenerating = true;
        
        // Simulate live updates
        const updateInterval = setInterval(() => {
            if (!this.isGenerating) {
                clearInterval(updateInterval);
                return;
            }

            // Update progress values with realistic changes
            this.demoData.collection.overall_progress = Math.min(1.0, 
                this.demoData.collection.overall_progress + Math.random() * 0.02);
            
            this.demoData.training.progress = Math.min(1.0,
                this.demoData.training.progress + Math.random() * 0.01);

            this.demoData.collection.total_items_collected += Math.floor(Math.random() * 10);

            // Randomly update system metrics
            this.demoData.system.cpu_usage = Math.max(20, Math.min(90, 
                this.demoData.system.cpu_usage + (Math.random() - 0.5) * 10));
            
            this.demoData.system.memory_usage = Math.max(30, Math.min(85,
                this.demoData.system.memory_usage + (Math.random() - 0.5) * 5));

            // Update dashboard if it exists
            if (window.dashboard) {
                window.dashboard.updateUI(this.demoData);
            }
        }, 2000);
    }

    stopDemo() {
        this.isGenerating = false;
    }

    getData() {
        return this.demoData;
    }
}

// Initialize charts when loaded
window.TrainingCharts = new TrainingCharts();

// Demo mode for development
if (window.location.search.includes('demo=true')) {
    window.demoGenerator = new DemoDataGenerator();
    
    // Override fetch to return demo data
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        if (url.includes('/training-api/status')) {
            return Promise.resolve({
                ok: true,
                json: () => Promise.resolve(window.demoGenerator.getData())
            });
        }
        
        if (url.includes('/training-api/trigger-')) {
            return Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ success: true })
            });
        }
        
        return originalFetch.apply(this, arguments);
    };
    
    // Start demo
    setTimeout(() => {
        window.demoGenerator.startDemo();
    }, 1000);
}