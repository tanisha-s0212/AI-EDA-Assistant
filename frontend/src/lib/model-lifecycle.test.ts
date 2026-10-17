import { beforeEach, describe, expect, it } from 'vitest';
import { useAppStore } from '@/lib/store';
import { clearStaleModelState } from '@/lib/model-lifecycle';

describe('clearStaleModelState', () => {
  beforeEach(() => {
    useAppStore.setState({
      modelId: 'old-model',
      modelTrained: true,
      selectedModel: 'Ridge',
      predictionResult: 42,
      predictionAnalysis: 'old result',
      predictionProbabilities: { yes: 0.8 },
      predictionHistory: [{ id: '1', prediction: 42, features: { signal: 3 }, timestamp: '2026-01-01' }],
      uploadedModel: { name: 'old-model', type: 'ridge', target: 'target', problem: 'regression', trainedAt: '', metrics: {}, features: [] },
    });
  });

  it('clears model and predictions when the backend rejects a stale artifact', () => {
    expect(clearStaleModelState({ response: { status: 409 } })).toBe(true);
    const state = useAppStore.getState();
    expect(state.modelId).toBeNull();
    expect(state.modelTrained).toBe(false);
    expect(state.predictionResult).toBeNull();
    expect(state.predictionHistory).toEqual([]);
    expect(state.uploadedModel).toBeNull();
  });

  it('leaves the active model alone for unrelated errors', () => {
    expect(clearStaleModelState({ response: { status: 500 } })).toBe(false);
    expect(useAppStore.getState().modelId).toBe('old-model');
  });
});
