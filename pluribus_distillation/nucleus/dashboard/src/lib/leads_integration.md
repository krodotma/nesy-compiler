# STRpLeadsView Integration Guide

## 1. Add to Dashboard Routes

In `/nucleus/dashboard/src/routes/index.tsx`, add the following:

### Import the component and types

```typescript
import { STRpLeadsView } from '../components/views/STRpLeadsView';
import type { STRpLead } from '../lib/state/leads_types';
import { LEAD_BUS_TOPICS } from '../lib/state/leads_types';
```

### Add state signal

```typescript
// After existing signals (around line 106)
const leads = useSignal<STRpLead[]>([]);
```

### Add to activeView type

```typescript
// Update the activeView type to include 'leads'
const activeView = useSignal<
  | 'home'
  | 'studio'
  | 'bus'
  | 'events'
  | 'agents'
  | 'requests'
  | 'sota'
  | 'semops'
  | 'services'
  | 'rhizome'
  | 'git'
  | 'types'
  | 'terminal'
  | 'plurichat'
  | 'webllm'
  | 'voice'
  | 'distill'
  | 'diagnostics'
  | 'browser-auth'
  | 'generative'
  | 'dkin'
  | 'metatest'
  | 'leads'  // <-- ADD THIS
>(initialView as any);
```

### Add to view allowlist

```typescript
// In the initialView IIFE, add 'leads' to allowed array
const allowed = [
  'home',
  'studio',
  'bus',
  'events',
  'agents',
  'requests',
  'sota',
  'semops',
  'services',
  'rhizome',
  'git',
  'types',
  'terminal',
  'plurichat',
  'webllm',
  'voice',
  'distill',
  'diagnostics',
  'generative',
  'dkin',
  'metatest',
  'leads',  // <-- ADD THIS
] as const;
```

### Handle bus events for leads

```typescript
// In the BroadcastChannel handler for 'pluribus-shadow'
case 'leads':
  leads.value = Array.isArray(payload.data) ? payload.data : [];
  break;
```

### Add view rendering

```typescript
// In the main view switch/conditional rendering
{activeView.value === 'leads' && (
  <STRpLeadsView
    leads={leads}
    dispatchAction={dispatchAction}
    connected={connected.value}
  />
)}
```

### Add to BicameralNav

In the navigation component, add:

```typescript
{
  id: 'leads',
  label: 'Leads',
  icon: '📋',
  badge: leads.value.filter(l => l.decision === 'promote').length,
}
```

---

## 2. Bus Event Topics

### Incoming Events (subscribe)

| Topic | Description | Payload |
|-------|-------------|---------|
| `strp.leads.sync` | Full leads list update | `{ leads: STRpLead[] }` |
| `strp.lead.created` | New lead created | `STRpLead` |
| `strp.lead.updated` | Lead modified | `STRpLead` |
| `strp.lead.deleted` | Lead removed | `{ lead_id: string }` |

### Outgoing Events (emit)

| Topic | Description | Payload |
|-------|-------------|---------|
| `strp.lead.action.watch` | Open content viewer | `{ lead_id, action, lead }` |
| `strp.lead.action.ingest` | Ingest to PORTAL | `{ lead_id, action, lead }` |
| `strp.lead.action.archive` | Archive lead | `{ lead_id, action, lead }` |
| `strp.lead.action.decision` | Change decision | `{ lead_id, action: 'promote'|'defer'|'reject', lead }` |

---

## 3. Backend Handler Example

```python
# In agent_bus.py handlers or dedicated leads_handler.py

from pluribus_next.tools.agent_bus import publish

async def handle_lead_action(event: dict) -> None:
    """Handle lead action events from dashboard."""
    data = event.get('data', {})
    action = data.get('action')
    lead_id = data.get('lead_id')
    lead = data.get('lead', {})

    if action == 'watch':
        # Open in browser or media player
        url = lead.get('url')
        await open_content_viewer(url)

    elif action == 'ingest':
        # Ingest to PORTAL
        await ingest_to_portal(lead)
        # Emit update
        lead['ingested'] = True
        publish(
            topic='strp.lead.updated',
            kind='event',
            level='info',
            actor='leads-handler',
            data=lead
        )

    elif action == 'archive':
        # Archive the lead
        await archive_lead(lead_id)
        lead['archived'] = True
        publish(
            topic='strp.lead.updated',
            kind='event',
            level='info',
            actor='leads-handler',
            data=lead
        )

    elif action in ('promote', 'defer', 'reject'):
        # Update decision
        lead['decision'] = action
        await update_lead_decision(lead_id, action)
        publish(
            topic='strp.lead.updated',
            kind='event',
            level='info',
            actor='leads-handler',
            data=lead
        )
```

---

## 4. Shadow Worker Integration

Add to shadow worker to sync leads from storage:

```typescript
// In shadow-worker.ts or equivalent

async function syncLeads(): Promise<void> {
  try {
    const response = await fetch('/api/leads');
    const leads = await response.json();

    const channel = new BroadcastChannel('pluribus-shadow');
    channel.postMessage({
      type: 'DATA_UPDATE',
      key: 'leads',
      data: leads,
    });
    channel.close();
  } catch (err) {
    console.error('Failed to sync leads:', err);
  }
}

// Add to polling loop
setInterval(syncLeads, 30000);
```

---

## 5. API Endpoint (if needed)

```python
# In api/routes.py or equivalent

@app.get('/api/leads')
async def get_leads():
    """Fetch all leads from storage."""
    leads_path = Path('.pluribus/leads/leads.json')
    if leads_path.exists():
        return json.loads(leads_path.read_text())
    return []

@app.post('/api/leads/{lead_id}/action')
async def lead_action(lead_id: str, action: LeadAction):
    """Perform action on a lead."""
    # Emit to bus
    await bus.emit(
        topic=f'strp.lead.action.{action.action}',
        kind='request',
        level='info',
        actor='api',
        data={
            'lead_id': lead_id,
            'action': action.action,
            'metadata': action.metadata,
        }
    )
    return {'status': 'queued'}
```

---

## 6. Type Export

Add to the main types export in `/lib/state/types.ts`:

```typescript
export type {
  STRpLead,
  LeadDecision,
  LeadTopic,
  LeadArtifacts,
  LeadTab,
  LeadSortBy,
  LeadFilterState,
  LeadAction,
  LeadActionRequest,
} from './leads_types';

export {
  LEAD_BUS_TOPICS,
  getDecisionColor,
  getTopicIcon,
  groupLeadsByDecision,
} from './leads_types';
```
