import { useAppStore } from '@/lib/store';

export function clearStaleModelState(error: unknown): boolean {
  const status = (error as { response?: { status?: number } } | null)?.response?.status;
  if (status !== 409) return false;
  useAppStore.setState({
    modelId: null,
    modelTrained: false,
    selectedModel: null,
    predictionResult: null,
    predictionAnalysis: null,
    predictionProbabilities: null,
    predictionHistory: [],
    uploadedModel: null,
  });
  return true;
}
