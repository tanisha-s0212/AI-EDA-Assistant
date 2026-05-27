import '@testing-library/jest-dom/vitest';

Element.prototype.scrollTo = Element.prototype.scrollTo ?? function scrollTo() {};
