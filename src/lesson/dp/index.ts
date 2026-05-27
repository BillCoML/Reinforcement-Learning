import type { Section } from '../section';
import { dpMeta } from './meta';
import { dpSection01 } from './section-01-setup';
import { dpSection02 } from './section-02-iterative-pe';
import { dpSection03 } from './section-03-policy-improvement';
import { dpSection04 } from './section-04-policy-iteration';
import { dpSection05 } from './section-05-value-iteration';
import { dpSection06 } from './section-06-async-dp';
import { dpSection07 } from './section-07-gpi';
import { dpSection08 } from './section-08-forward-links';

export const dpSections: Section[] = [
  dpSection01,
  dpSection02,
  dpSection03,
  dpSection04,
  dpSection05,
  dpSection06,
  dpSection07,
  dpSection08,
];

export { dpMeta };
