/**
 * Isotope Manager (Graph-Native)
 * Handles layout and rendering of graph nodes as tiles
 */
export class IsotopeManager {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        this.isotope = null;
    }

    init() {
        if (!this.container) return;
        
        console.log('[Isotope] Initializing...');
        
        // @ts-ignore
        this.isotope = new Isotope(this.container, {
            itemSelector: '.document-tile',
            layoutMode: 'masonry',
            masonry: {
                columnWidth: '.grid-sizer',
                gutter: '.gutter-sizer'
            },
            transitionDuration: '0.4s'
        });

        // Emergency Fix: Force layout
        setTimeout(() => this.isotope.layout(), 500);
    }

    render(documents) {
        if (!documents || documents.length === 0) return;

        console.log(`[Isotope] Rendering ${documents.length} items`);
        
        const items = documents.map(doc => this.createTile(doc));
        this.container.append(...items);
        this.isotope.appended(items);
        this.isotope.layout();
    }

    createTile(doc) {
        const div = document.createElement('div');
        div.className = `document-tile cluster-${doc.cluster}`;
        div.innerHTML = `
            <div class="tile-inner">
                <h3>${doc.title}</h3>
                <p>${doc.summary}</p>
                <div class="meta">
                    <span class="badge">${doc.type}</span>
                </div>
            </div>
        `;
        return div;
    }
}
