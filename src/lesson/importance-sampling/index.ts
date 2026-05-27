import type { Section } from '../section';
import { isMeta } from './meta';
import { isSection01 } from './section-01-is-identity';
import { isSection02 } from './section-02-variance';
import { isSection03 } from './section-03-trajectory-is';
import { isSection04 } from './section-04-per-decision';
import { isSection05 } from './section-05-forward-links';

export const isSections: Section[] = [
  isSection01,
  isSection02,
  isSection03,
  isSection04,
  isSection05,
];

export { isMeta };
