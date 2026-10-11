"""Fixed TP/PP latency from Paper v2 Eqs. (3)-(4) and the text after (7b).

``compute_delay_seconds_per_token`` stores ``d_comp[i,j,k]`` and
``communication_delay_seconds_per_token`` stores ``d_comm[i,j,k]``.
Both coefficients are in seconds/token; returned delays are in seconds.
"""

from llm_resource_allocation.domain import ParallelismConfig, QueryType


def compute_ttft_seconds(
    compute_delay_seconds_per_token: float,
    query: QueryType,
    config: ParallelismConfig,
) -> float:
    """Return time to first token: ``D_TTFT = d_comp[i,j,k]*h_i/n`` (v2 Eq. 3).

    ``query.input_tokens`` is ``h_i``; ``config.tensor_parallel_degree`` is ``n``.
    The supplied compute coefficient is the TP=1 baseline. Output is in seconds.
    """

    return (
        compute_delay_seconds_per_token
        * query.input_tokens
        / config.tensor_parallel_degree
    )


def compute_generation_delay_seconds(
    compute_delay_seconds_per_token: float,
    communication_delay_seconds_per_token: float,
    query: QueryType,
    config: ParallelismConfig,
) -> float:
    """Return generation delay: ``D_Gen = (d_comp/n + m*d_comm)*f_i`` (v2 Eq. 4).

    ``query.output_tokens`` is ``f_i``; ``config.tensor_parallel_degree`` is ``n``
    and ``config.pipeline_parallel_depth`` is ``m``. Output is in seconds.
    """

    per_output_token_delay = (
        compute_delay_seconds_per_token / config.tensor_parallel_degree
        + config.pipeline_parallel_depth
        * communication_delay_seconds_per_token
    )
    return per_output_token_delay * query.output_tokens


def compute_configuration_delay_seconds(
    compute_delay_seconds_per_token: float,
    communication_delay_seconds_per_token: float,
    query: QueryType,
    config: ParallelismConfig,
) -> float:
    """Return ``D_cfg = D_TTFT + D_Gen`` in seconds (v2 Eqs. 3-4).

    Sum ``compute_ttft_seconds`` and ``compute_generation_delay_seconds``.
    This equals ``d_comp*(h_i+f_i)/n + m*d_comm*f_i`` (v2, after Eq. 7b),
    with h_i and f_i from ``query.input_tokens`` and ``query.output_tokens``;
    n and m are ``config.tensor_parallel_degree`` and
    ``config.pipeline_parallel_depth``.
    """

    ttft = compute_ttft_seconds(
        compute_delay_seconds_per_token,
        query,
        config,
    )
    generation_delay = compute_generation_delay_seconds(
        compute_delay_seconds_per_token,
        communication_delay_seconds_per_token,
        query,
        config,
    )
    return ttft + generation_delay
