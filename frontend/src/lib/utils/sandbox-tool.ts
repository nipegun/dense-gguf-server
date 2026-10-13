import { fTranslate } from '$lib/i18n';
import { cSandboxTimeoutMsDefault, cSandboxTimeoutMsMax, cSandboxToolName } from '$lib/constants';
import { JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

const cNerdamerDescription = `
Symbolic/numeric math via \`nerdamer\`
nerdamer(expr,subs?,opts?)/nerdamer.func(...)→Expression Format via .text(fmt?) (fmt: 'decimals'|'fractions'|'scientific') eval via .evaluate(subs?)
nerdamer(expr,{x:2}) substitutes numeric via opts 'numer' or .evaluate()
simplify/expand/factor(expr) div/gcd/lcm(...) coeffs/partfrac(expr,var)
diff/integrate(expr,var) defint(expr,lo,hi,var?) sum/product(expr,var,lo,hi) limit(expr,var,pt)
solve(expr,var) solveEquations([eq1,eq2],[var1,var2])
polarform/rectform/arg/realpart/imagpart(z)
set/get Var/Constant(name,val?) setFunction(name,[params],body)
IMPORTANT:Identifier 'nerdamer' has already been declared, use it directly`;

/**
 * Build the sandbox tool definition. When `includeSymbolicMath` is true,
 * the description includes nerdamer API documentation; otherwise it
 * describes a plain JavaScript sandbox.
 */
export function fBuildSandboxToolDefinition(pIncludeSymbolicMath: boolean): OpenAIToolDefinition {
  return {
    function: {
      description: pIncludeSymbolicMath
        ? `Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.${cNerdamerDescription}`
        : 'Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.',
      name: cSandboxToolName,
      parameters: {
        properties: {
          code: {
            description: fTranslate('text_8b54c4c622a2'),
            type: JsonSchemaType.cString
          },
          timeout_ms: {
            description: `Execution timeout in milliseconds, default ${cSandboxTimeoutMsDefault}, max ${cSandboxTimeoutMsMax}`,
            type: JsonSchemaType.cNumber
          }
        },
        required: ['code'],
        type: JsonSchemaType.cObject
      }
    },
    type: ToolCallType.cFunction
  };
}
