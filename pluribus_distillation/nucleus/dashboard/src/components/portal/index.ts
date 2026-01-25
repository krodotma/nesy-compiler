/**
 * portal Components - A2UI Entry Point
 *
 * portal (Pluribus Organism Runtime Thresholds and Logistics)
 * is the robust entry-point for world-model ingestion.
 *
 * @see nucleus/specs/portal_contract_v1.md
 * @see nucleus/plans/portal_a2ui_implementation.md
 */

// Core selector for ingress mode
export {
  PortalIngressSelector,
  type PortalIngressSelectorProps,
  type IngressMode,
  type ConfidenceScore,
  type DestinationPath,
  type IngressFragment,
} from './PortalIngressSelector';

export {
  PortalDestinationSelector,
  type PortalDestinationSelectorProps,
  type IngressDestination,
  type IngressDestinationType,
  type IngressSelection,
} from './PortalDestinationSelector';

export { PortalIngressPanel } from './PortalIngressPanel';
export { PortalIngestDropzone } from './PortalIngestDropzone';
export { PortalInceptionPanel } from './PortalInceptionPanel';

export { default } from './PortalIngressSelector';
