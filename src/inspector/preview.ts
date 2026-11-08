import type { RemoteObject } from './jsc-types';

//const MAX_PREVIEW_PROPS = 5;

/** Stringify a JSC value field: primitives show themselves, anything else falls back. */
function remoteValueText(v: unknown, fallback: string): string {
    switch (typeof v) {
        case 'string':
            return v;
        case 'number':
        case 'boolean':
        case 'bigint':
            return String(v);
        default:
            return fallback;
    }
}


/**
 * Render a JSC RemoteObject to a readable DAP Variable value string.
 * Primitives show their value (strings quoted), objects/functions their
 * description, with a small preview tail for collections.
 */
export function formatRemoteObject(obj: RemoteObject | undefined): string {
    // Getter-only accessors come back without a `value`; JSC omits it entirely.
    if (obj === undefined) return '(getter)';
    switch (obj.type) {
        case 'undefined':
            return 'undefined';
        case 'string':
            return `'${String(obj.value)}'`;
        case 'boolean':
        case 'number':
        case 'bigint':
            // Unserializable values (bigint, NaN, ±Infinity) carry no `value`;
            // JSC always provides the printable form in `description`.
            return remoteValueText(obj.value, obj.description ?? obj.type);
        case 'symbol':
            return obj.description ?? 'Symbol()';
        case 'function': {
            if (obj.description) {
                const name = /^function\s+([^(\s]*)/.exec(obj.description)?.[1] ?? '';
                if (name) {
                    return `\u0192 ${name}`;
                }
                if (obj.className) {
                    return `\u0192 ${obj.className}`;
                }
                // @ts-expect-error ignore-TS2532
                const truncated = obj.description.split('\n')[0].slice(0, 80);
                return `\u0192 ${truncated}`;
            }
            return `\u0192 ${obj.className ?? 'anonymous'}`;
        }
        case 'object': {
            if (obj.subtype === 'null') return 'null';
            let base = obj.preview?.description ?? obj.className ?? obj.description ?? 'Object';
            if (base.length > 80) {
                base = base.slice(0, 80) + '…';
            }
            return base ;
        }
        default:
            return obj.description ?? remoteValueText(obj.value, obj.type);
    }
}
