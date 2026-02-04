/**
 * Living Brain - Main Entry Point
 */
import { ApiClient } from './api-client.js';
import { IsotopeManager } from './isotope-manager.js';

class App {
    constructor() {
        this.api = new ApiClient();
        this.ui = new IsotopeManager('#document-list');
    }

    async start() {
        console.log('[App] Starting Living Brain...');
        
        // 1. Initialize UI
        this.ui.init();

        // 2. Fetch Data (Graph)
        const docs = await this.api.getDocuments(50); // Start small

        // 3. Render
        this.ui.render(docs);
    }
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window.app.start();
});
