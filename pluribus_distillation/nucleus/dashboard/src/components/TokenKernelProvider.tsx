import {
  component$,
  createContextId,
  useContextProvider,
  useContext,
  useVisibleTask$,
  Slot,
  noSerialize,
  type NoSerialize,
} from '@builder.io/qwik';
import { getTokenGeometryClient, type TokenGeometryClient } from '../lib/token-geometry-client';

export interface TokenKernelContextValue {
  client: NoSerialize<TokenGeometryClient>;
}

export const TokenKernelContext = createContextId<TokenKernelContextValue>('token-kernel-context');

export const TokenKernelProvider = component$(() => {
  const client = noSerialize(getTokenGeometryClient());
  useContextProvider(TokenKernelContext, { client });

  useVisibleTask$(() => {
    (window as any).__tokenKernel = client;
  });

  return <Slot />;
});

export const useTokenKernel = () => useContext(TokenKernelContext);
