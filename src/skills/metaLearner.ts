export async function fetchMetaSuggestion(payload: any): Promise<any> {
    const resp = await fetch('http://localhost:8001/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error('MetaLearner request failed');
    return await resp.json();
}
