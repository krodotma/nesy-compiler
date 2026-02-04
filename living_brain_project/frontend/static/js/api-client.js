/**
 * API Client for Living Brain
 * connects to FalkorDB (via proxy) and Spine Registry
 */
export class ApiClient {
    constructor(baseUrl = '/api/v1') {
        this.baseUrl = baseUrl;
    }

    async getDocuments(limit = 100, offset = 0) {
        // In a real scenario, this would query FalkorDB via a backend proxy
        // For now, we simulate the graph query structure
        console.log(`[ApiClient] Fetching documents limit=${limit} offset=${offset}`);
        
        try {
            // This endpoint should be shimmed by the python backend
            const response = await fetch(`${this.baseUrl}/graph/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: `MATCH (d:Document) RETURN d SKIP ${offset} LIMIT ${limit}`
                })
            });
            
            if (!response.ok) throw new Error('Graph query failed');
            return await response.json();
        } catch (e) {
            console.warn('[ApiClient] Fallback to mock data (Graph not ready)');
            return this.getMockData(limit);
        }
    }

    getMockData(limit) {
        return Array.from({ length: limit }, (_, i) => ({
            id: `doc-${i}`,
            title: `Document ${i}`,
            type: 'concept',
            summary: 'This is a placeholder document from the Living Brain.',
            cluster: i % 5
        }));
    }
}
