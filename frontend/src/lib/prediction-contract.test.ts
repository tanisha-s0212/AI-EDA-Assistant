import { describe, expect, it } from 'vitest';
import { predictionInputs } from './prediction-contract';

describe('prediction input contract', () => {
  it('preserves decimal measurements and categorical identifiers', () => {
    expect(predictionInputs([
      { name: 'amount', role: 'numeric', value: '12.3456' },
      { name: 'customer', role: 'categorical', value: '0012' },
    ])).toEqual({ amount: 12.3456, customer: '0012' });
  });
  it.each(['12oops', 'Infinity', 'NaN', '', ' '])('rejects malformed numeric input %s', value => {
    expect(() => predictionInputs([{ name: 'amount', role: 'numeric', value }])).toThrow();
  });
});
